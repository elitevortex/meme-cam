"""Meme cam: do a pose in front of your webcam and a GIF plays over your face.

Run:   .venv/bin/python meme_cam.py
Keys:  q = quit, d = toggle debug numbers, c = recalibrate your straight face

Left and right are YOUR left and right. The picture is mirrored, so your left
side is on the left of the screen.
"""

import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageSequence

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Settings you'll probably want to tweak. Turn on debug (press d) to see the
# live values, then nudge these up or down until the poses trigger reliably.
# ---------------------------------------------------------------------------
CALIBRATION_SECONDS = 2  # at startup, keep a straight face this long so it learns your neutral face
EYES_SHUT = 0.15         # speed: how much MORE closed your eyes must be than your straight face
LIPS_PUCKER = 0.12       # speed: how much MORE pushed-out your lips must be than your straight face
HEAD_TURN = 0.2          # dog/cat: how far your nose is off-centre (0 = facing camera)
ARM_RAISE = 0.03         # broccoli: how far above your left shoulder your left hand must be
MOUTH_REACH = 0.45       # durag: how close a hand must be to your mouth, in face widths
HOLD_FRAMES = 4          # a pose must be seen this many frames in a row to start
RELEASE_SECONDS = 0.4    # keep the GIF up this long after you drop the pose
GIF_SIZE = 0.5           # GIF width as a fraction of the camera width (0.5 = half the camera)
FLIPPED = {"hold_on"}    # GIFs to mirror, so the meme matches your side
PLAY_FULL = {"broccoli"}  # GIFs that play to the end once triggered, even if you drop the pose

# Pose name -> GIF file. Order matters: the first pose that matches wins.
POSES = {
    "broccoli": "broccoli.gif",        # left hand up over your left shoulder
    "durag": "durag.gif",              # pull your shirt up to your mouth (a hand at your mouth)
    "hold_on": "hold_on.gif",          # right fist held up
    "speed": "speed.gif",              # eyes squeezed shut + lips pushed out
    "dog_side_eye": "dog_side_eye.gif",  # look left
    "cat": "cat.gif",                  # look right
}


# ---------------------------------------------------------------------------
# GIF playback
# ---------------------------------------------------------------------------
class Gif:
    def __init__(self, path, flip=False):
        img = Image.open(path)
        self.frames, self.durations = [], []
        for frame in ImageSequence.Iterator(img):
            rgba = np.array(frame.convert("RGBA"))
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            self.frames.append(cv2.flip(bgra, 1) if flip else bgra)
            self.durations.append(max(frame.info.get("duration", 80), 20) / 1000)
        self.total = sum(self.durations)
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def current_frame(self):
        t = (time.time() - self.start) % self.total
        for frame, d in zip(self.frames, self.durations):
            if t < d:
                return frame
            t -= d
        return self.frames[-1]


def paste(dst, src_bgra, cx, cy, width):
    """Draw src centred on (cx, cy) at the given width, clipped to the frame."""
    h, w = src_bgra.shape[:2]
    height = int(width * h / w)
    src = cv2.resize(src_bgra, (width, height))
    x0, y0 = cx - width // 2, cy - height // 2
    H, W = dst.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + width), min(H, y0 + height)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    patch = src[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255
    region = dst[dy0:dy1, dx0:dx1]
    region[:] = (patch[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pose detection
# ---------------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def is_fist(hand):
    """Every fingertip (index to pinky) is closer to the wrist than its middle joint."""
    wrist = hand[0]
    return all(dist(hand[tip], wrist) < dist(hand[pip], wrist) * 1.1
               for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))


class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y


def face_geometry(face, body):
    """Mouth, nose, chin and face width, from the face model or (if your face is covered) the body model."""
    if face:
        cheeks = sorted((face[234], face[454]), key=lambda p: p.x)
        mouth = Point((face[13].x + face[14].x) / 2, (face[13].y + face[14].y) / 2)
        return mouth, face[1], face[152].y, cheeks[1].x - cheeks[0].x
    if body:
        # Body model points: 0 = nose, 9/10 = mouth corners, 11/12 = shoulders
        mouth = Point((body[9].x + body[10].x) / 2, (body[9].y + body[10].y) / 2)
        width = abs(body[11].x - body[12].x) * 0.45
        return mouth, body[0], mouth.y + width * 0.5, width
    return None


def hand_over_left_shoulder(body, hands):
    """Is a hand up above shoulder height, over on your left (the left of the mirrored screen)?"""
    left_sh, right_sh = sorted((body[11], body[12]), key=lambda p: p.x)
    middle_x = (left_sh.x + right_sh.x) / 2

    def over_left_shoulder(x, y):
        return x < middle_x and left_sh.y - y > ARM_RAISE

    # Hand model is the most accurate; fall back to the body model's wrists if it misses the hand
    tracked = sum(over_left_shoulder(sum(p.x for p in h) / len(h), sum(p.y for p in h) / len(h)) for h in hands)
    wrists = sum(w.visibility > 0.5 and over_left_shoulder(w.x, w.y) for w in (body[15], body[16]))
    return max(tracked, wrists) >= 1


def eyes_score(blend):
    """How closed/squeezed both eyes are (0-1): the stronger of blinking or squinting."""
    return max(min(blend["eyeBlinkLeft"], blend["eyeBlinkRight"]),
               min(blend["eyeSquintLeft"], blend["eyeSquintRight"]))


def lips_score(blend):
    """How pushed-out your lips are (0-1): a kiss-face pucker or an 'ooh' funnel."""
    return max(blend["mouthPucker"], blend["mouthFunnel"])


def detect_pose(face, blend, hands, body, baseline):
    """Return the name of the first matching pose, or None. Also returns debug values."""
    info = {}
    if body and hand_over_left_shoulder(body, hands):
        return "broccoli", info

    geo = face_geometry(face, body)

    if geo:
        mouth, nose, chin_y, face_w = geo
        # Durag: some part of a hand is right at your mouth, and the hand isn't up above your nose
        for hand in hands:
            reach = min(dist(p, mouth) for p in hand) / face_w
            info["mouth_reach"] = min(info.get("mouth_reach", 9.0), reach)
            centre_y = sum(p.y for p in hand) / len(hand)
            if reach < MOUTH_REACH and centre_y > nose.y:
                return "durag", info

        # Hold on: a fist on your right (right side of the mirrored screen), held up above your chin
        for hand in hands:
            if is_fist(hand) and hand[0].x > nose.x and hand[0].y < chin_y:
                return "hold_on", info

    if not face:
        return None, info

    # Face poses, measured against your calibrated straight face
    eyes = eyes_score(blend) - baseline["eyes"]
    pucker = lips_score(blend) - baseline["pucker"]
    cheeks = sorted((face[234], face[454]), key=lambda p: p.x)
    turn = (face[1].x - cheeks[0].x) / max(cheeks[1].x - cheeks[0].x, 1e-6) - 0.5
    info.update(eyes=eyes, pucker=pucker, turn=turn)

    if eyes > EYES_SHUT and pucker > LIPS_PUCKER:
        return "speed", info
    if turn < -HEAD_TURN:   # nose towards the left of the mirrored screen = you're looking left
        return "dog_side_eye", info
    if turn > HEAD_TURN:
        return "cat", info
    return None, info


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    gifs = {name: Gif(HERE / "gifs" / file, flip=name in FLIPPED) for name, file in POSES.items()}

    def model(name):
        return BaseOptions(model_asset_path=str(HERE / "models" / name))

    face_detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=model("face_landmarker.task"), running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True, num_faces=1,
    ))
    hand_detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=model("hand_landmarker.task"), running_mode=vision.RunningMode.VIDEO, num_hands=2,
    ))
    body_detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=model("pose_landmarker_lite.task"), running_mode=vision.RunningMode.VIDEO, num_poses=1,
    ))

    # Resizable window, so tiling window managers (AeroSpace etc.) can size it without it collapsing
    cv2.namedWindow("Meme Cam", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Meme Cam", 960, 540)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise SystemExit("Couldn't open the webcam. On macOS, allow camera access for your terminal "
                         "in System Settings > Privacy & Security > Camera, then restart the terminal.")

    active, candidate, streak, last_seen = None, None, 0, 0.0
    face_box = None  # (cx, cy, width) in pixels, remembered so the GIF stays put if the face is lost
    debug = False
    t0, ts = time.monotonic(), -1
    calib_samples, calib_until = [], time.time() + CALIBRATION_SECONDS
    baseline = {"eyes": 0.0, "pucker": 0.0}

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror, so it feels like a mirror
        H, W = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = max(ts + 1, int((time.monotonic() - t0) * 1000))  # MediaPipe needs every frame to be later than the last

        face_res = face_detector.detect_for_video(image, ts)
        hand_res = hand_detector.detect_for_video(image, ts)
        body_res = body_detector.detect_for_video(image, ts)

        face = face_res.face_landmarks[0] if face_res.face_landmarks else None
        blend = ({b.category_name: b.score for b in face_res.face_blendshapes[0]}
                 if face_res.face_blendshapes else {})
        body = body_res.pose_landmarks[0] if body_res.pose_landmarks else None
        if face:
            xs, ys = [p.x for p in face], [p.y for p in face]
            face_box = (int((min(xs) + max(xs)) / 2 * W), int((min(ys) + max(ys)) / 2 * H),
                        int((max(xs) - min(xs)) * W))

        now = time.time()
        if now < calib_until:
            # Learn what your straight face looks like before triggering anything
            if blend:
                calib_samples.append((eyes_score(blend), lips_score(blend)))
            cv2.putText(frame, "Calibrating: keep a straight face", (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            pose, info = None, {}
        else:
            if calib_samples:
                baseline = {"eyes": float(np.mean([s[0] for s in calib_samples])),
                            "pucker": float(np.mean([s[1] for s in calib_samples]))}
                calib_samples = []
            pose, info = detect_pose(face, blend, hand_res.hand_landmarks, body, baseline)

        # Only switch once a pose has been held a few frames, and hang on briefly after it ends
        streak = streak + 1 if pose and pose == candidate else 1
        candidate = pose
        # GIFs in PLAY_FULL play all the way through once started, even if you drop the pose
        locked = active in PLAY_FULL and now - gifs[active].start < gifs[active].total
        if locked:
            if pose == active:
                last_seen = now
        elif pose and streak >= HOLD_FRAMES:
            if pose != active:
                gifs[pose].restart()
            active, last_seen = pose, now
        elif active and pose == active:
            last_seen = now
        elif active and now - last_seen > RELEASE_SECONDS:
            active = None

        if active and face_box:
            cx, cy, _ = face_box
            paste(frame, gifs[active].current_frame(), cx, cy, int(W * GIF_SIZE))

        if debug:
            lines = [f"pose: {pose}  showing: {active}",
                     f"straight face: eyes {baseline['eyes']:.2f} pucker {baseline['pucker']:.2f}"] + [
                f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in info.items()]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (10, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Meme Cam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug = not debug
        if key == ord("c"):
            calib_samples, calib_until = [], time.time() + CALIBRATION_SECONDS

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
