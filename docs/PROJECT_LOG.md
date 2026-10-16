# Project Log: Decisions, Problems and Fixes

This is the build story of the football analytics pipeline: what I did at each step, what went
wrong, and how I fixed it. The README describes what the project does. This file explains how it
got there.

**Final pipeline:** detection → pitch filtering → tracking → team classification → camera-motion
compensation → pitch mapping → ID merging → stats and minimap

## At a glance

| # | Step | Main problem | Fix |
|---|------|--------------|-----|
| 1 | Choose footage | Broadcast feeds cut to close-ups and replays | Use one continuous tactical-camera clip |
| 2 | Load the clip in Colab | `FPS: 0.0`, `Frames loaded: 0` | Wrong file path; checked the path and the file's existence |
| 3 | Detect players | 26 to 34 detections, but only about 22 to 25 people on the pitch | Filter by grass colour and pitch region |
| 4 | Pitch region | Fixed polygon broke as the camera panned | Per-frame pitch mask from the largest green area |
| 5 | Team colours | 4 clusters mixed teams with staff | 6 clusters, contact sheet, manual cluster-to-team labels |
| 6 | Tracking | Single-frame colour errors, staff counted as players | ByteTrack IDs plus a per-ID majority vote |
| 7 | Pitch mapping | Camera moves, only part of the pitch is visible | Homography from landmarks, chained with optical flow |
| 8 | Broken IDs | Players re-appeared with new IDs | Merge by team, timing and pitch position |
| 9 | Speed and distance | A 45 km/h "top speed" | Gaps in paths; interpolate, smooth, 1-second window |
| 10 | Results and demo | Heatmaps look like trails | Be honest: 10 seconds is too short; built a side-by-side demo |
| 11 | Publishing | Push rejected ("fetch first") | Pulled and merged the remote's commits, then pushed |

---

## 1. Scoping the project

**Goal:** turn match video into player data, and go beyond a heatmap.

**What I did:** listed what tracking data makes possible: heatmaps, distance and speed, possession,
passes, team shape (compactness, defensive line), a top-down minimap. I put them in a build order:
detection and tracking, team colours, possession, pitch mapping, then heatmaps, speed, passes and a
dashboard.

**Decision:** treat pitch mapping (homography) as the key upgrade. It is what turns a demo into a
project that can measure real distance and speed, and many copies of this project skip it.

## 2. Choosing the footage

**Problem:** normal TV broadcast footage cuts between cameras (close-ups, replays), which breaks
tracking and pitch mapping.

**Decision:** use a continuous tactical-camera clip (elevated, wide, no cuts). The Real Madrid vs
Barcelona clip fit this well: kits are easy to tell apart and the pitch lines are clear.

**Challenges I expected from this angle:**
- players are only about 30 to 40 pixels tall, so the detector can miss some
- the ball is a few pixels wide and almost invisible
- the camera pans and zooms, so screen position is not pitch position
- players leave the frame, so stats only cover the visible region
- staff, photographers and the crowd get detected too
- YouTube compression blurs small players

**Rule I set for myself:** call all results "visible-region stats" and never claim full-match totals.

## 3. Loading the clip in Colab

**Problem:** the first frame-loading block printed `FPS: 0.0`, `Size: 0 x 0`, `Frames loaded: 0`.

**Cause:** OpenCV could not open the file. Colab runs on a remote machine, so a path on my laptop
does not exist there, and the path I typed did not match the uploaded file.

**Fix:** upload the clip to Colab (or mount Google Drive), copy the exact path, and check it with
`os.path.exists` and a folder listing before loading. If the file exists but still fails to open,
convert it to a standard mp4 with ffmpeg.

## 4. First detection test

**What I did:** ran YOLO11x on five frames, people class only, `conf=0.2`, `imgsz=1280` (a larger
input size helps with small players). The counts were 34, 34, 26, 28 and 28.

**Problem:** more people than players on the pitch. The extras were staff, photographers and
officials along the touchline.

**Decision:** detection itself was working, so I did not need tiling. The job was filtering.

## 5. Filtering to the pitch

### 5a. Grass check under the feet

I kept a detection only if the pixels at the bottom of its box were mostly green (HSV range
35 to 85 hue). Counts fell to 26, 29, 25, 26 and 27, but staff standing on grass still got through.

### 5b. Fixed pitch polygon (a dead end)

I drew a polygon around the pitch. Two mistakes:
- **Wrong scale.** I estimated corners from a shrunken screenshot, so the outline was too small and
  would have deleted real players. Fix: draw a labelled pixel grid on the real 1920x1080 frame and
  read coordinates from it.
- **The camera moves.** Even with correct corners, a fixed polygon only fits one frame. As the
  camera pans, the real pitch edges move and the polygon stays put.

### 5c. Per-frame pitch mask

**Fix:** find green pixels in every frame, close the gaps left by players and lines, and keep the
largest green region as the pitch. It follows the camera automatically.

**Remaining issue:** staff on the thin grass strip beside the touchline are still inside the mask.
Counts were 28, 28, 25, 26 and 26. I stopped trying to fix this with geometry and handled it with
shirt colour (next section).

**Lesson:** any filter built from fixed pixel coordinates fails on a moving camera.

## 6. Team classification

**What I did:** took the median colour of each player's chest area (green pixels removed), in LAB
colour space, and clustered the colours with k-means.

**Problem 1: 4 clusters were not enough.** Madrid (white) was clean, but Barcelona's dark kit was
split across two clusters and mixed with staff in dark clothes. The referee's yellow ended up in a
Barcelona cluster.

**Fix:** use 6 clusters, then draw a contact sheet (a row of player crops for each cluster) and
label the clusters by eye:

| Cluster | Content | Label |
|---|---|---|
| 0 | goalkeeper, referees (yellow) | other |
| 1 | Barcelona, plus one staff member | barca |
| 2 | Real Madrid | madrid |
| 3 | Barcelona, plus a referee | barca |
| 4 | staff in dark jackets | other |
| 5 | mixed, blurry | other |

**Problem 2: swapped colours.** I applied an example label map instead of the one I read off the
contact sheet, so Barcelona and Madrid showed in the wrong colours. Rerunning with the correct map
fixed it. I also checked every result visually against the footage.

**Lesson:** single-frame colour guesses are noisy, so a person who looks different in one frame is
mislabelled. That is solved by tracking (next section).

## 7. Tracking and the per-ID vote

**What I did:** ran ByteTrack on 300 frames (10 seconds), and for every tracked ID collected the
team guess from every frame. Each ID takes its most common guess, and IDs seen in fewer than
15 frames are dropped.

**Result:** 41 raw IDs. After the vote and the filter: Barcelona 15, Madrid 11, other 7, with 8
short-lived IDs dropped.

**Review (frames at 0, 3, 6 and 9.5 seconds):**
- IDs stayed on the same players even as the camera panned from the goal to the centre circle
- referees, goalkeeper and touchline staff were correctly "other"
- two staff members were still labelled Barcelona (the top-edge official and a bench-area staff
  member). I set those two IDs to "other" by hand and noted it in the README
- some players had new IDs late in the clip, which explains the extra Barcelona and Madrid IDs

**Minor:** a message about the `lap` package installing itself appeared. It did not need a runtime
restart, and restarting would have wiped everything in memory.

## 8. Pitch mapping (homography)

**Goal:** convert pixel positions to metres on a pitch, so distance and speed can be measured.

**Step 1: calibrate on the first frame.** I matched six visible pitch-line points (penalty-box
corners, six-yard-box corners, halfway line ends) to their real positions in a 105 x 68 m pitch,
and computed a homography. Warping the frame showed the boxes and centre circle lining up with the
drawn lines. The black empty area was expected, because one frame never shows the whole pitch.

**Caveat:** I placed the landmark points by eye, so positions are accurate to about 1 to 2 m.

**Step 2: follow the camera.** Each frame gets its own mapping by tracking grass and line features
between consecutive frames (optical flow), ignoring the player boxes, and chaining each small shift
onto the previous mapping.

**Check:** a top-down minimap at frames 0, 150 and 290. Staff standing still on the touchline moved
by about 2 m or less over the whole clip, which showed the camera tracking held up.

**Limitation:** because small shifts are chained, errors add up. Fine for 10 seconds, but a longer
clip needs re-anchoring to the pitch lines every few seconds.

## 9. Merging broken IDs

**Problem:** three players had left the frame or been lost for a moment and came back with new IDs,
which inflated the counts (Barcelona 13, Madrid 11 after the manual staff fix).

**Rule v1:** two IDs are the same player if they are the same team, the new one starts after the old
one ends (within 3 seconds), and they are within 6 m. It only merged two chains of Barcelona IDs.

**Diagnosis:** for the two pairs it missed (9 and 183, 17 and 135) I printed the gap and the distance:
- 9 and 183: gap of 0 frames, distance 0.2 m
- 17 and 135: gap of -4 frames (they overlap), distance 0.2 m

They were obviously the same players, and my rule had required the gap to be greater than zero.

**Fix:** allow up to 10 frames of overlap. Final counts: Barcelona 10, Madrid 10, other 9. That is
exactly 10 outfield players per team.

**Caveat:** a looser rule can merge two different players standing close together. The same-team
requirement and the 6 m limit reduce that risk, and I checked that every merge was a pair I
recognised.

## 10. Distance and speed

**Problem:** the first stats table showed a 45.4 km/h top speed for one player (and 26.5 for
another), when everyone else was jogging.

**Cause:** gaps in a player's path. When the detector misses a player for a few frames, two
consecutive recorded positions were many frames apart, but I treated them as one frame apart. A
normal 3 m move over 10 frames looked like a 3 m move in a single frame.

**Fix:**
1. fill every missing frame by interpolating the path
2. smooth the path (moving average over 15 frames)
3. measure top speed as the distance moved over a one-second window, not a single step

**Result:** the same player dropped from 45.4 to 13.4 km/h. Top speeds across all players ended up
between about 6 and 17 km/h and distances between 12 and 31 m, which are sensible for jogging pace
over 10 seconds.

## 11. Heatmaps, average positions and the demo

**What I found:**
- The heatmaps look like short trails. Ten seconds is not enough for meaningful zones.
- The average-position plot showed both teams mixed together in the middle of the pitch, not two
  separate groups. I had expected two groups, and that expectation was wrong. Both teams were
  marking each other, and the plot only covers the slice of pitch the camera was showing, so it says
  nothing about formation.

**Decision:** describe these as a working pipeline to be run on a longer clip, not as findings.

**Demo:** I built a side-by-side video with the tracked footage on the left and the live minimap on
the right, and converted it to a GIF for the README. This shows the pipeline working better than
any single plot.

## 12. Publishing to GitHub

**Repo hygiene:**
- rewrote the README to match the new pipeline (demo GIF first, results, challenges, limitations)
- fixed repo paths (the notebook lives in `notebooks/`)
- expanded `requirements.txt` (scikit-learn, scipy, pandas, matplotlib, lap) and `.gitignore`
- kept model weights (`*.pt`, about 109 MB, over GitHub's file limit) and match footage (copyrighted)
  out of the repo

**Problem:** `git push` was rejected with "fetch first".

**Cause:** the repo on GitHub had a commit my local copy did not have, not the username change I
suspected. GitHub redirects renamed accounts.

**Fix:** update the remote URL to the new username, pull and merge the remote commits, check
`git status` for large files, then push.

---

## Key takeaways

- **Fixed pixel coordinates fail on a moving camera.** Use per-frame masks or a mapping that
  follows the camera.
- **One frame is noisy; a vote across a track is not.** Persistent IDs let small errors get outvoted.
- **Diagnose before loosening a rule.** Printing the gap and distance for the two missed merges
  showed exactly which condition was wrong, so I fixed that one and did not loosen everything.
- **An impossible number means a bug.** The 45 km/h top speed was a data-gap bug, not a sprint.
- **State the limits.** 10 seconds, a visible region, hand-placed landmarks, and two manual overrides.

## What is next

- re-anchor the pitch mapping to the pitch lines automatically, for longer clips
- run on a 2 to 3 minute clip so heatmaps and average positions mean something
- pass detection and a pass network (needs the ball, which is the least reliable part at this
  resolution)
- a Streamlit dashboard
