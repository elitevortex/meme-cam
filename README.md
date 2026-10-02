# Meme Cam

Strike a pose in front of your webcam and a meme GIF plays over your face.

Meme Cam tracks your face, hands and body in real time with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide), works out which pose you're making with a few simple rules, and pastes the matching GIF over your face. It runs entirely on your machine. No training, no cloud, no GPU needed.

## Requirements

- macOS (tested on Apple Silicon). Linux and Windows should work, but are untested.
- Python 3.10 or newer
- A webcam

## Setup

```bash
git clone https://github.com/elitevortex/meme-cam.git
cd meme-cam
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The tracking models are included in `models/`, so there's nothing else to download.

## Running it

```bash
.venv/bin/python meme_cam.py
```

On macOS, the first run asks for permission to use the camera. Allow it, then run the command again.

**Keep a straight face for the first 2 seconds.** Meme Cam uses that time to learn what your neutral face looks like, so face poses trigger on how *your* face changes rather than on an average face.

### Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `l` | Show / hide the tracking lines |
| `d` | Show / hide live detection values (for tuning) |
| `c` | Recalibrate your straight face |

## Poses

Left and right mean *your* left and right. The picture is mirrored, so your left side is on the left of the screen.

| Pose | How to do it | GIF |
|---|---|---|
| Broccoli | Lift your left hand up over your left shoulder | `broccoli.gif` |
| Durag | Pull your shirt up to your mouth | `durag.gif` |
| Hold on | Hold your right fist up above your chin | `hold_on.gif` |
| Speed | Squeeze your eyes shut and push your lips out | `speed.gif` |
| Side-eye | Turn your head to look left | `dog_side_eye.gif` |
| Cat | Turn your head to look right | `cat.gif` |

A GIF appears once a pose has been held for a few frames, and disappears shortly after you stop. Broccoli always plays to the end once it starts.

With the tracking lines on, you'll see your hand and body skeletons plus the trigger zones: a line above your left shoulder for broccoli and a line at your chin for hold on. A zone turns red while you're triggering it.

## Tuning

If a pose triggers too easily or not at all, press `d` to see the live values, then adjust the settings at the top of `meme_cam.py`:

| Setting | Controls |
|---|---|
| `EYES_SHUT`, `LIPS_PUCKER` | How much your eyes and lips must change from your straight face for Speed |
| `HEAD_TURN` | How far you must turn your head for side-eye and cat |
| `ARM_RAISE` | How far above your shoulder your hand must go for broccoli |
| `MOUTH_REACH` | How close your hand must be to your mouth for durag |
| `HOLD_FRAMES`, `RELEASE_SECONDS` | How quickly GIFs appear and disappear |
| `GIF_SIZE` | GIF width as a fraction of the camera width |
| `FLIPPED` | GIFs to mirror horizontally |
| `PLAY_FULL` | GIFs that always play to the end once triggered |

## Adding your own pose

1. Drop a GIF into `gifs/`.
2. Add an entry to `POSES` in `meme_cam.py`. Order matters: the first pose that matches wins.
3. Add a rule for it in `detect_pose()`. The face model gives you 478 face points plus expression scores such as `jawOpen` and `browInnerUp`. The hand model gives 21 points per hand, and the body model gives 33 body points.

## How it works

Every frame:

1. **Capture.** OpenCV grabs a webcam frame and mirrors it.
2. **Track.** Three MediaPipe models find your face (478 points plus about 50 expression scores), your hands (21 points each) and your body (33 points).
3. **Classify.** Plain geometry rules turn the points into a pose. A fist means every fingertip is closer to the wrist than its middle knuckle. Looking left means your nose sits well left of the middle of your face.
4. **Smooth.** A pose must hold for a few frames before its GIF appears, which stops it flickering on and off.
5. **Draw.** The current GIF frame is scaled and pasted over your face, and follows you as you move.

## Troubleshooting

**"Couldn't open the webcam".** Allow camera access for your terminal app in System Settings > Privacy & Security > Camera, then restart the terminal.

**The window collapses to a tiny title bar.** A tiling window manager such as AeroSpace or yabai is resizing it. Make Python windows float instead. For AeroSpace, add this to `aerospace.toml`:

```toml
[[on-window-detected]]
if.app-id = "org.python.python"
run = "layout floating"
```

**It crashes on startup with `Check failed: service_ Service is unavailable`.** You're on MediaPipe 1.0.x, which crashes on macOS when loading the face model. Install the pinned version with `pip install -r requirements.txt`.

**The face poses trigger when you aren't doing anything.** Press `c` and hold a straight face for 2 seconds to recalibrate.

## Credits

The GIFs come from [Giphy](https://giphy.com) and belong to their respective owners. They're included for personal, non-commercial use only.
