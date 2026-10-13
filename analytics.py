"""Analytics on the CSVs written by track.py: ball possession and a team heatmap.

Usage:
    python analytics.py --video clip.mp4 --team "Real Madrid"
"""
import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEAM_SCORE = {"Real Madrid": 1, "Barcelona": -1}   # +1 / -1 so we can average over time


def possession(positions_csv, ball_csv, max_dist=120, out_png="possession.png"):
    """Credit the ball to the closest player (within max_dist pixels) in each frame."""
    pos = pd.read_csv(positions_csv)
    ball = pd.read_csv(ball_csv)
    players = pos[pos.team.isin(TEAM_SCORE)]
    vals = pd.Series(np.nan, index=range(int(ball.frame.max()) + 1))

    for f, bx, by in ball.itertuples(index=False):
        p = players[players.frame == f]
        if p.empty:
            continue
        d = np.hypot(p.x - bx, p.y - by)
        if d.min() < max_dist:
            vals[f] = TEAM_SCORE[p.loc[d.idxmin(), "team"]]

    smooth = vals.rolling(30, min_periods=1, center=True).mean()   # about 1 second
    a, b = (smooth > 0).sum(), (smooth < 0).sum()
    total = max(a + b, 1)
    print(f"Ball + a nearby player found in {vals.notna().sum()} of {len(vals)} frames")
    print(f"Real Madrid: {100 * a / total:.0f}%   Barcelona: {100 * b / total:.0f}%")

    fig, ax = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={"height_ratios": [1, 1.2]})
    ax[0].barh(["Possession"], [a / total * 100], color="lightgrey", edgecolor="k", label="Real Madrid")
    ax[0].barh(["Possession"], [b / total * 100], left=[a / total * 100], color="royalblue", label="Barcelona")
    ax[0].set_xlim(0, 100)
    ax[0].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.6))
    ax[1].imshow([np.sign(smooth.fillna(0)).values], aspect="auto", cmap="coolwarm_r", vmin=-1, vmax=1)
    ax[1].set_yticks([])
    ax[1].set_xlabel("Frame (blue = Barcelona, red = Real Madrid, white = nobody)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


def heatmap(positions_csv, video, team, out_png="heatmap.png"):
    """Paint a team's positions over the first video frame (screen coordinates)."""
    d = pd.read_csv(positions_csv)
    d = d[d.team == team]
    cap = cv2.VideoCapture(video)
    ok, bg = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read video: {video}")
    if d.empty:
        raise SystemExit(f"No rows for team '{team}' in {positions_csv}")
    h, w = bg.shape[:2]

    heat = np.zeros((h, w), np.float32)
    for x, y in zip(d.x, d.y):
        if 0 <= x < w and 0 <= y < h:
            heat[int(y), int(x)] += 1
    heat = cv2.GaussianBlur(heat, (0, 0), 25)
    heat = heat / max(heat.max(), 1e-9)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (heat > 0.05)[..., None]
    overlay = np.where(mask, cv2.addWeighted(bg, 0.5, color, 0.5, 0), bg)
    cv2.imwrite(out_png, overlay)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--team", default="Real Madrid", choices=list(TEAM_SCORE))
    ap.add_argument("--positions", default="positions.csv")
    ap.add_argument("--ball", default="ball.csv")
    a = ap.parse_args()
    possession(a.positions, a.ball)
    heatmap(a.positions, a.video, a.team)
