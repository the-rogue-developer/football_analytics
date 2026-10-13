"""Football tracking pipeline.

Detect players and ball (YOLO11) -> track with persistent IDs (ByteTrack)
-> classify team by shirt colour (HSV) -> draw overlays -> save positions to CSV.

Usage:
    python track.py --video clip.mp4
"""
import argparse
from collections import Counter, deque

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Team names and colour rules below are tuned for ONE clip
# (Real Madrid in white, Barcelona in blue/dark-red stripes, yellow referee,
# neon-green goalkeeper). Check them on a frame of your own footage first.
TEAM_COLORS = {  # BGR drawing colours
    "Barcelona": (255, 0, 0),
    "Real Madrid": (255, 255, 255),
    "Referee": (0, 255, 255),
    "GK": (0, 255, 0),
}


def get_team(frame, box):
    """Guess the team from the shirt colour in the upper body of a player box."""
    x1, y1, x2, y2 = map(int, box)
    torso = frame[y1 + (y2 - y1) // 5: y1 + (y2 - y1) // 2, x1:x2]
    if torso.size == 0:
        return "Unknown"
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    n = torso.shape[0] * torso.shape[1]

    def share(mask):
        return cv2.countNonZero(mask) / n

    yellow = share(cv2.inRange(hsv, (20, 100, 120), (35, 255, 255)))
    gk = share(cv2.inRange(hsv, (50, 100, 180), (75, 255, 255)))  # neon green, not grass
    white = share(cv2.inRange(hsv, (0, 0, 170), (180, 50, 255)))
    blue = share(cv2.inRange(hsv, (100, 80, 40), (130, 255, 255)))
    red = share(cv2.inRange(hsv, (0, 80, 40), (10, 255, 255))) + \
        share(cv2.inRange(hsv, (160, 80, 40), (180, 255, 255)))

    if yellow > 0.12:
        return "Referee"
    if gk > 0.15:
        return "GK"
    if white > 0.25:
        return "Real Madrid"
    if blue > 0.45:          # solid navy suit = staff, not striped shirt
        return "Unknown"
    if blue + red > 0.06:
        return "Barcelona"
    return "Unknown"


def on_pitch(frame, box):
    """True if the area around the player's feet is mostly green grass."""
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) // 2
    patch = frame[max(y2 - 6, 0): y2 + 6, max(cx - 15, 0): cx + 15]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)).mean() / 255 > 0.4


def run(video, out_video, model_name, tracker_cfg, imgsz):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5) or 30
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    model = YOLO(model_name)

    team_votes, ball_trail, lost = {}, deque(maxlen=25), 0
    records, ball_records, frame_no = [], [], 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, tracker=tracker_cfg, classes=[0, 32],
                          conf=0.1, imgsz=imgsz, verbose=False)[0]

        best_ball = None
        for box in res.boxes:
            cls, conf = int(box.cls[0]), float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == 32:                                   # sports ball
                if best_ball is None or conf > best_ball[0]:
                    best_ball = (conf, (x1 + x2) // 2, (y1 + y2) // 2)
                continue
            if conf < 0.3 or box.id is None or not on_pitch(frame, (x1, y1, x2, y2)):
                continue
            tid = int(box.id[0])
            team_votes.setdefault(tid, deque(maxlen=30)).append(
                get_team(frame, (x1, y1, x2, y2)))
            team = Counter(team_votes[tid]).most_common(1)[0][0]   # smoothed over 30 frames
            if team == "Unknown":
                continue
            color = TEAM_COLORS[team]
            cx = (x1 + x2) // 2
            records.append((frame_no, tid, team, cx, y2))          # feet position
            cv2.ellipse(frame, (cx, y2), (max((x2 - x1) // 2, 8), 6), 0, -45, 235, color, 2)
            cv2.putText(frame, f"#{tid}", (cx - 10, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if best_ball:
            bx, by = best_ball[1], best_ball[2]
            if not ball_trail or np.hypot(bx - ball_trail[-1][0], by - ball_trail[-1][1]) < 150:
                ball_trail.append((bx, by))
                lost = 0
                ball_records.append((frame_no, bx, by))
            else:                                           # sudden jump = probably a false hit
                lost += 1
                if lost > 10:
                    ball_trail.clear()
        else:
            lost += 1
            if lost > 5 and ball_trail:
                ball_trail.popleft()
        for i, p in enumerate(ball_trail):
            cv2.circle(frame, p, 2 + i // 5, (0, 200, 255), -1)

        writer.write(frame)
        frame_no += 1
        if frame_no % 50 == 0:
            print(f"Processed {frame_no} frames")

    cap.release()
    writer.release()
    pd.DataFrame(records, columns=["frame", "id", "team", "x", "y"]).to_csv("positions.csv", index=False)
    pd.DataFrame(ball_records, columns=["frame", "x", "y"]).to_csv("ball.csv", index=False)
    print(f"Done: {frame_no} frames, {len(records)} player rows, {len(ball_records)} ball rows")
    print(f"Saved {out_video}, positions.csv, ball.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Football player and ball tracking")
    ap.add_argument("--video", required=True, help="input match clip (mp4)")
    ap.add_argument("--out", default="output.mp4", help="annotated output video")
    ap.add_argument("--model", default="yolo11l.pt", help="YOLO weights (downloaded automatically)")
    ap.add_argument("--tracker", default="bytetrack_football.yaml")
    ap.add_argument("--imgsz", type=int, default=1280, help="detection resolution")
    a = ap.parse_args()
    run(a.video, a.out, a.model, a.tracker, a.imgsz)
