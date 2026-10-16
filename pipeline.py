"""
pipeline.py - football tracking pipeline that survives Colab disconnects.

Everything slow (detection, tracking, camera motion) runs in chunks and is saved to
disk after each chunk. If Colab disconnects, run the processing step again and it
continues from the last saved chunk.

Typical use (see the notebook):
    import pipeline as pl
    cfg = pl.make_config(...)                      # clip, start, length, stride, chunk
    pl.run(cfg, run_dir, H0, kmeans, team_map)     # resumable
    data = pl.load_run(run_dir)                    # after it finishes
"""
import glob
import json
import os
import pickle
import shutil
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

__version__ = "1.0"

SESSION_STEP = 100000        # id offset per Colab session (tracker ids restart after a disconnect)


# --------------------------------------------------------------------------
# small file helpers (writes are atomic, so a disconnect mid-write can't corrupt a file)
# --------------------------------------------------------------------------
def _atomic_write(path, writer):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


def save_json(obj, path):
    _atomic_write(path, lambda f: f.write(json.dumps(obj, indent=2).encode()))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_pickle(obj, path):
    _atomic_write(path, lambda f: pickle.dump(obj, f))


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def make_config(drive_clip, local_clip="/content/clip.mp4", start=0, n_frames=1800, stride=1,
                chunk=300, conf=0.2, imgsz=1280, tracker="bytetrack.yaml"):
    """start/n_frames are in ORIGINAL video frames. stride=1 uses every frame."""
    cap = cv2.VideoCapture(drive_clip if not os.path.exists(local_clip) else local_clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return dict(drive_clip=drive_clip, clip=local_clip, start=int(start), n_frames=int(n_frames),
                stride=int(stride), chunk=int(chunk), conf=conf, imgsz=int(imgsz),
                tracker=tracker, fps=float(fps), total_frames=total)


def ensure_local_clip(cfg):
    """Copy the clip from Drive to Colab's fast local disk (re-done after every disconnect)."""
    if not os.path.exists(cfg["clip"]):
        shutil.copy(cfg["drive_clip"], cfg["clip"])
    return cfg["clip"]


def read_frame(cfg, index):
    """Read one frame by its index in the original video."""
    cap = cv2.VideoCapture(cfg["clip"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
_MODEL = None


def get_model(weights="yolo11x.pt"):
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(weights)
    return _MODEL


# --------------------------------------------------------------------------
# vision building blocks (the same ones that worked in the prototype notebook)
# --------------------------------------------------------------------------
def pitch_mask(frame):
    """Largest green region of the frame = the pitch. Follows the camera automatically."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(green)
    if cnts:
        biggest = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask, [biggest], -1, 255, -1)
    return mask


def in_pitch_dynamic(mask, box):
    x1, y1, x2, y2 = box
    fx = int(min(max((x1 + x2) / 2, 0), mask.shape[1] - 1))
    fy = int(min(max(y2, 0), mask.shape[0] - 1))
    return mask[fy, fx] > 0


def torso_color(frame, box):
    """Median chest colour (LAB), ignoring green pixels. None if the box is too small."""
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(x1, 0), max(y1, 0)
    h, w = y2 - y1, x2 - x1
    if h < 4 or w < 3:
        return None
    crop = frame[y1 + int(0.2 * h): y1 + int(0.55 * h), x1 + int(0.2 * w): x2 - int(0.2 * w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    notgreen = ~((hsv[..., 0] > 35) & (hsv[..., 0] < 85) & (hsv[..., 1] > 40))
    pix = crop[notgreen]
    if len(pix) < 5:
        pix = crop.reshape(-1, 3)
    lab = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, axis=0)


def camera_step(prev_gray, gray, prev_mask, prev_boxes):
    """3x3 matrix mapping this frame's pixels to the previous frame's pixels (ignores people)."""
    m = prev_mask.copy()
    for x1, y1, x2, y2 in prev_boxes:
        cv2.rectangle(m, (int(x1) - 8, int(y1) - 8), (int(x2) + 8, int(y2) + 8), 0, -1)
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=800, qualityLevel=0.01,
                                 minDistance=10, mask=m)
    P = np.eye(3)
    if p0 is not None and len(p0) >= 8:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
        good = st.ravel() == 1
        if good.sum() >= 8:
            Pm, _ = cv2.findHomography(p1[good], p0[good], cv2.RANSAC, 3.0)
            if Pm is not None:
                P = Pm
    return P


def calibrate(img_pts, pitch_pts):
    """Homography from first-frame pixels to pitch metres. Needs 4+ points, not all on a line."""
    img_pts = np.array(img_pts, dtype=np.float32)
    pitch_pts = np.array(pitch_pts, dtype=np.float32)
    if len(img_pts) < 4 or len(img_pts) != len(pitch_pts):
        raise ValueError("Need the same number of image and pitch points, at least 4.")
    H, _ = cv2.findHomography(img_pts, pitch_pts, cv2.RANSAC)
    if H is None:
        raise ValueError("Could not compute a mapping. Check that the points are not on one line.")
    return H


# --------------------------------------------------------------------------
# team colours
# --------------------------------------------------------------------------
def collect_color_samples(cfg, n_frames=14, log=print):
    """Detect people on evenly spaced frames; return their chest colours and small crops."""
    model = get_model()
    n_proc = cfg["n_frames"] // cfg["stride"]
    picks = np.linspace(0, max(n_proc - 1, 0), n_frames).astype(int)
    colors, crops = [], []
    for k in picks:
        frame = read_frame(cfg, cfg["start"] + int(k) * cfg["stride"])
        if frame is None:
            continue
        mask = pitch_mask(frame)
        res = model(frame, classes=[0], conf=cfg["conf"], imgsz=cfg["imgsz"], verbose=False)[0]
        for b in res.boxes.xyxy.cpu().numpy():
            if not in_pitch_dynamic(mask, b):
                continue
            c = torso_color(frame, b)
            if c is None:
                continue
            x1, y1, x2, y2 = map(int, b)
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            if crop.size == 0:
                continue
            colors.append(c)
            crops.append(cv2.resize(crop, (50, 110)))
    log(f"Collected {len(colors)} player samples from {len(picks)} frames")
    return np.array(colors), crops


def contact_sheet(crops, labels, k, per_row=14):
    rows = []
    for c in range(k):
        sel = [cr for cr, l in zip(crops, labels) if l == c][:per_row]
        while len(sel) < per_row:
            sel.append(np.zeros((110, 50, 3), np.uint8))
        row = np.hstack(sel)
        cv2.putText(row, f"cluster {c}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows.append(row)
    return np.vstack(rows)


def check_team_map(team_map, k):
    allowed = {"barca", "madrid", "other"}
    bad = [c for c in range(k) if str(c) not in team_map and c not in team_map]
    if bad:
        raise ValueError(f"team_map is missing clusters {bad}")
    out = {}
    for c in range(k):
        v = team_map.get(c, team_map.get(str(c)))
        if v not in allowed:
            raise ValueError(f"Cluster {c} is '{v}'. Use one of {sorted(allowed)}.")
        out[str(c)] = v
    return out


# --------------------------------------------------------------------------
# the long, resumable run
# --------------------------------------------------------------------------
def _checkpoints(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "checkpoints", "ckpt_*.pkl")))


def progress(cfg, run_dir):
    n_proc = cfg["n_frames"] // cfg["stride"]
    files = _checkpoints(run_dir)
    done = load_pickle(files[-1])["state"]["f_next"] if files else 0
    return done, n_proc


def run(cfg, run_dir, H0, kmeans, team_map, log=print):
    """Process the clip in chunks. Safe to call again after a disconnect: it resumes."""
    ensure_local_clip(cfg)
    model = get_model()
    stride, chunk = cfg["stride"], cfg["chunk"]
    n_proc = cfg["n_frames"] // stride
    team_map = {str(k): v for k, v in team_map.items()}

    files = _checkpoints(run_dir)
    state = load_pickle(files[-1])["state"] if files else None
    f = state["f_next"] if state else 0
    session = (state["session"] + 1) if state else 0
    if f >= n_proc:
        log(f"Already finished ({f}/{n_proc} frames). Nothing to do.")
        return
    log(f"Starting at frame {f}/{n_proc} (session {session})"
        + (" - resuming from the last saved chunk" if state else ""))

    cap = cv2.VideoCapture(cfg["clip"])
    prev_gray = prev_mask = None
    prev_boxes = []
    if state:
        # rebuild the previous frame so camera tracking continues without a jump
        cap.set(cv2.CAP_PROP_POS_FRAMES, cfg["start"] + (f - 1) * stride)
        ok, pf = cap.read()
        if ok:
            prev_gray = cv2.cvtColor(pf, cv2.COLOR_BGR2GRAY)
            prev_mask = pitch_mask(pf)
            prev_boxes = state["prev_boxes"]
            for _ in range(stride - 1):
                cap.grab()
        Hcur = np.array(state["Hcur"], dtype=np.float64)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cfg["start"])
        Hcur = np.array(H0, dtype=np.float64)

    buf_rows, buf_H = {}, {}
    t0, f_at_t0 = time.time(), f

    def flush():
        if not buf_rows:
            return
        first = min(buf_rows)
        state_out = dict(f_next=f, Hcur=Hcur.tolist(), session=session,
                         prev_boxes=np.array(prev_boxes).tolist())
        save_pickle(dict(frames=dict(buf_rows), Hs=dict(buf_H), state=state_out),
                    os.path.join(run_dir, "checkpoints", f"ckpt_{first:06d}.pkl"))
        rate = (f - f_at_t0) / max(time.time() - t0, 1e-6)
        left = (n_proc - f) / max(rate, 1e-6) / 60
        log(f"saved up to frame {f}/{n_proc}  |  {rate:.1f} frames/s  |  about {left:.0f} min left")
        buf_rows.clear()
        buf_H.clear()

    while f < n_proc:
        ok, frame = cap.read()
        if not ok:
            break
        for _ in range(stride - 1):
            cap.grab()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = pitch_mask(frame)
        if prev_gray is not None:
            Hcur = Hcur @ camera_step(prev_gray, gray, prev_mask, prev_boxes)

        res = model.track(frame, persist=True, tracker=cfg["tracker"], classes=[0],
                          conf=cfg["conf"], imgsz=cfg["imgsz"], verbose=False)[0]
        boxes_all = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None

        rows = []
        if ids is not None:
            keep, colors = [], []
            for b, tid in zip(boxes_all, ids):
                if not in_pitch_dynamic(mask, b):
                    continue
                c = torso_color(frame, b)
                if c is None:
                    continue
                keep.append((int(tid), b))
                colors.append(c)
            if keep:
                labs = kmeans.predict(np.array(colors))
                feet = np.array([[((b[0] + b[2]) / 2, b[3])] for _, b in keep], dtype=np.float32)
                pts = cv2.perspectiveTransform(feet, Hcur)[:, 0, :]
                for (tid, b), lab, (px, py) in zip(keep, labs, pts):
                    rows.append((tid + session * SESSION_STEP, float(b[0]), float(b[1]),
                                 float(b[2]), float(b[3]), team_map[str(int(lab))],
                                 float(px), float(py)))
        buf_rows[f] = rows
        buf_H[f] = Hcur.copy()
        prev_gray, prev_mask, prev_boxes = gray, mask, boxes_all
        f += 1
        if len(buf_rows) >= chunk:
            flush()
    flush()
    cap.release()
    log("Finished." if f >= n_proc else f"Video ended early at frame {f}.")


def load_run(run_dir):
    """Merge every saved chunk. Returns frames {f: rows}, Hs {f: 3x3}.
    A row is (id, x1, y1, x2, y2, team_guess, pitch_x, pitch_y)."""
    frames, Hs = {}, {}
    for p in _checkpoints(run_dir):
        d = load_pickle(p)
        frames.update(d["frames"])
        Hs.update(d["Hs"])
    return dict(frames=frames, Hs=Hs)


# --------------------------------------------------------------------------
# after the run: team vote, ID merging, trajectories, stats
# --------------------------------------------------------------------------
def vote_teams(frames, min_frames=15):
    votes = defaultdict(list)
    for rows in frames.values():
        for r in rows:
            votes[r[0]].append(r[5])
    id_team = {t: Counter(v).most_common(1)[0][0] for t, v in votes.items() if len(v) >= min_frames}
    return id_team, len(votes) - len(id_team)


def trajectories(frames, ids=None):
    traj = defaultdict(dict)
    for f, rows in frames.items():
        for r in rows:
            if ids is None or r[0] in ids:
                traj[r[0]][f] = (r[6], r[7])
    return traj


def merge_ids(traj, id_team, max_dist=6.0, max_gap=90, max_overlap=10):
    """Join IDs that are the same player: same team, new starts as the old ends, close together."""
    ids = [t for t in id_team if id_team[t] != "other" and t in traj]
    span = {t: (min(traj[t]), max(traj[t])) for t in ids}
    ids.sort(key=lambda t: span[t][0])
    merge, used = {}, set()
    for new in ids:
        s_new = span[new][0]
        p_new = np.array(traj[new][s_new])
        best, best_d = None, max_dist
        for old in ids:
            if old == new or old in used or id_team[old] != id_team[new]:
                continue
            s_old, e_old = span[old]
            if s_old >= s_new:
                continue
            if -max_overlap <= s_new - e_old <= max_gap:
                d = np.linalg.norm(p_new - np.array(traj[old][e_old]))
                if d < best_d:
                    best, best_d = old, d
        if best is not None:
            merge[new] = best
            used.add(best)
    return merge


def root_of(t, merge):
    while t in merge:
        t = merge[t]
    return t


def finalize(frames, id_team, merge, force_other=()):
    """Apply merges and manual overrides. Returns final_team {id: team}, final_traj {id: {f: (x, y)}}."""
    id_team = dict(id_team)
    for t in force_other:
        if t in id_team:
            id_team[t] = "other"
    traj = trajectories(frames, set(id_team))
    final_team, final_traj = {}, defaultdict(dict)
    for t in id_team:
        if t in traj:
            r = root_of(t, merge)
            final_team[r] = id_team[t]
            final_traj[r].update(traj[t])
    return final_team, final_traj


def describe_ids(final_team, final_traj, fps):
    """One line per player-ID, to spot staff and mistakes by eye."""
    out = []
    for t, path in final_traj.items():
        frs = sorted(path)
        xy = np.array([path[f] for f in frs])
        out.append(dict(id=t, team=final_team[t], first=frs[0], last=frs[-1],
                        seconds=round(len(frs) / fps, 1),
                        median_x=round(float(np.median(xy[:, 0])), 1),
                        median_y=round(float(np.median(xy[:, 1])), 1)))
    return sorted(out, key=lambda r: (r["team"], r["id"]))


def compute_stats(final_team, final_traj, fps, min_seconds=2.0):
    """Distance and top speed per player. Fills gaps, smooths, measures speed over 1 second."""
    import pandas as pd
    from scipy.ndimage import uniform_filter1d
    rows = []
    win = max(3, int(round(0.5 * fps)))
    one_sec = max(1, int(round(fps)))
    for pid, path in final_traj.items():
        if final_team[pid] == "other":
            continue
        frs = np.array(sorted(path))
        if len(frs) < min_seconds * fps:
            continue
        xy = np.array([path[f] for f in frs])
        full = np.arange(frs[0], frs[-1] + 1)
        x = np.interp(full, frs, xy[:, 0])
        y = np.interp(full, frs, xy[:, 1])
        xs = uniform_filter1d(x, win, mode="nearest")
        ys = uniform_filter1d(y, win, mode="nearest")
        step = np.hypot(np.diff(xs), np.diff(ys))
        if len(xs) > one_sec:
            disp = np.hypot(xs[one_sec:] - xs[:-one_sec], ys[one_sec:] - ys[:-one_sec])
            top = float(disp.max() * 3.6)
        else:
            top = 0.0
        rows.append(dict(player_id=pid, team=final_team[pid],
                         seconds_visible=round(len(frs) / fps, 1),
                         distance_m=round(float(step.sum()), 1), top_speed_kmh=round(top, 1)))
    return pd.DataFrame(rows).sort_values(["team", "player_id"]).reset_index(drop=True)
