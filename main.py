"""
Google TV Gesture Controller
=============================
Uses the NEW MediaPipe Tasks API (mp.tasks.vision.HandLandmarker).

HOW TO GESTURE:
  - Point your index finger and SWIPE in a direction  → LEFT / RIGHT / UP / DOWN
  - Pinch thumb + index finger together               → OK / SELECT
  - The OTHER fingers should ideally be curled in when swiping,
    but it works fine either way. The key is to make a clear, quick swipe
    with just your index fingertip as the "pointer."

Anti-return-swipe logic:
  After a swipe fires the history clears AND a cooldown blocks new triggers,
  so bringing your hand back will never double-fire.

Setup:
  pip install mediapipe opencv-contrib-python
  (hand_landmarker.task model is auto-downloaded ~10 MB on first run)

ADB (to actually control your TV):
  1. Enable Developer Options → ADB / Network Debugging on Google TV
  2. adb connect <TV_IP>:5555   OR plug in via USB
  3. Set ADB_HOST below to your TV's IP:port, or leave as None to just
     display gestures on screen without sending ADB commands.
"""

import os
import sys
import time
import urllib.request
from collections import deque
import os
import cv2
import mediapipe as mp

ADB = os.environ['ADB_HOST']  # Optional override via environment variable

# ── ADB config ─────────────────────────────────────────────────────────────────
ADB_HOST = ADB 

KEYEVENTS = {"LEFT": 21, "RIGHT": 22, "UP": 19, "DOWN": 20, "CENTER": 23}

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# ── Swipe tuning ───────────────────────────────────────────────────────────────
HISTORY_WINDOW_S     = 0.35   # seconds of index-tip history to analyse
MIN_SWIPE_DISTANCE   = 0.12   # normalised (0–1) displacement needed
MIN_SWIPE_VELOCITY   = 0.35   # normalised units/sec — filters accidental drifts
DIRECTIONALITY_RATIO = 0.70   # primary axis must dominate this much (0–1)
GESTURE_COOLDOWN_S   = 1.0    # seconds before another swipe can fire

# ── Thumbs-up / OK tuning ──────────────────────────────────────────────────────
# Thumb tip must be above the thumb base knuckle (MCP) by at least this amount.
THUMB_RISE_MIN       = 0.08   # normalised y difference (tip.y < mcp.y - this)
# Each non-thumb finger is considered curled when its tip is below its PIP joint.
THUMBS_UP_COOLDOWN_S = 1.0    # seconds before thumbs-up can fire again

# ── Index finger "pointing" check ──────────────────────────────────────────────
# We only track swipes when the index finger is extended in ANY direction.
# Uses Euclidean distance between tip (landmark 8) and MCP knuckle (landmark 5).
# A curled finger pulls the tip close to the knuckle; an extended finger doesn't.
# This works for pointing left, right, up, AND down — no directional bias.
INDEX_EXTEND_THRESHOLD = 0.14  # normalised tip-to-MCP distance = "extended"

# ── Hand skeleton connections ──────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX

GESTURE_LABELS = {
    "LEFT":   "◀  LEFT",
    "RIGHT":  "RIGHT  ▶",
    "UP":     "▲  UP",
    "DOWN":   "DOWN  ▼",
    "CENTER": "👍  OK",
}

GESTURE_COLORS = {
    "LEFT":   (80,  200, 255),
    "RIGHT":  (80,  200, 255),
    "UP":     (80,  255, 180),
    "DOWN":   (80,  255, 180),
    "CENTER": (80,  200, 255),
}


# ── Model download ─────────────────────────────────────────────────────────────
def ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand_landmarker.task (~10 MB) …", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("done.")
    except Exception as exc:
        print(f"\nFailed: {exc}")
        print(f"Download manually:\n  {MODEL_URL}\nand place at: {MODEL_PATH}")
        sys.exit(1)


# ── ADB helper ─────────────────────────────────────────────────────────────────
def send_adb_key(action: str) -> None:
    import subprocess
    base = ["adb"]
    if ADB_HOST:
        base += ["-s", ADB_HOST]
    base += ["shell", "input", "keyevent", str(KEYEVENTS[action])]
    subprocess.Popen(base, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── Gesture state machine ──────────────────────────────────────────────────────
class GestureState:
    def __init__(self) -> None:
        self.history: deque[tuple[float, float, float]] = deque()
        self.last_fire_time: float = 0.0
        self.last_thumbsup_time: float = 0.0   # separate cooldown for thumbs-up

    def push(self, x: float, y: float) -> None:
        now = time.time()
        self.history.append((now, x, y))
        cutoff = now - HISTORY_WINDOW_S
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def clear_history(self) -> None:
        self.history.clear()

    def check_swipe(self) -> str | None:
        if time.time() - self.last_fire_time < GESTURE_COOLDOWN_S:
            return None
        if len(self.history) < 5:
            return None

        t0, x0, y0 = self.history[0]
        t1, x1, y1 = self.history[-1]
        dt = t1 - t0
        if dt < 0.04:
            return None

        dx, dy = x1 - x0, y1 - y0
        dist = (dx ** 2 + dy ** 2) ** 0.5

        if dist < MIN_SWIPE_DISTANCE or dist / dt < MIN_SWIPE_VELOCITY:
            return None

        adx, ady = abs(dx), abs(dy)
        total = adx + ady
        if total == 0:
            return None

        if adx >= ady:
            if adx / total < DIRECTIONALITY_RATIO:
                return None
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            if ady / total < DIRECTIONALITY_RATIO:
                return None
            direction = "DOWN" if dy > 0 else "UP"

        # Fire — clear history so return journey can't retrigger
        self.last_fire_time = time.time()
        self.history.clear()
        return direction

    def check_thumbs_up(self, lms) -> bool:
        """Fires when thumb is raised and all other fingers are curled."""
        if time.time() - self.last_thumbsup_time < THUMBS_UP_COOLDOWN_S:
            return False
        # Thumb tip (4) must be well above thumb MCP (2) — y decreases upward
        thumb_tip = lms[4]
        thumb_mcp = lms[2]
        if not (thumb_mcp.y - thumb_tip.y > THUMB_RISE_MIN):
            return False
        # All four fingers must be curled: tip below PIP joint
        # (index=8/6, middle=12/10, ring=16/14, pinky=20/18)
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip_i, pip_i in finger_pairs:
            if lms[tip_i].y < lms[pip_i].y:   # tip is above PIP = finger extended
                return False
        self.last_thumbsup_time = time.time()
        return True


# ── Drawing helpers ────────────────────────────────────────────────────────────
def is_index_extended(lms) -> bool:
    """True if the index finger is extended in any direction.

    Measures the straight-line distance between the fingertip (landmark 8)
    and the MCP knuckle (landmark 5). A finger curled into a fist keeps the
    tip close to the knuckle regardless of hand orientation; an extended
    finger always pushes the tip far away — works for all four swipe directions.
    """
    tip = lms[8]   # INDEX_FINGER_TIP
    mcp = lms[5]   # INDEX_FINGER_MCP (base knuckle)
    dist = ((tip.x - mcp.x) ** 2 + (tip.y - mcp.y) ** 2) ** 0.5
    return dist > INDEX_EXTEND_THRESHOLD


def draw_hand(frame, lms, index_extended: bool) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (110, 110, 110), 1, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        # Highlight index finger brighter when extended
        if i in (5, 6, 7, 8) and index_extended:
            color  = (0, 230, 255)
            radius = 5 if i == 8 else 4
        else:
            color  = (80, 200, 80)
            radius = 4
        cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)


def draw_swipe_trail(frame, history: deque, h: int, w: int) -> None:
    """Draw a fading trail of the index finger's recent path."""
    pts = [(int(x * w), int(y * h)) for _, x, y in history]
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        color = (int(alpha * 0), int(alpha * 200), int(alpha * 255))
        cv2.line(frame, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)
    if pts:
        cv2.circle(frame, pts[-1], 6, (0, 230, 255), -1, cv2.LINE_AA)


def draw_hud(frame, state: "GestureState", last_gesture: str | None, cooldown_left: float,
             tu_cooldown: float, index_extended: bool) -> None:
    h, w = frame.shape[:2]

    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Cooldown bar (bottom of the top strip)
    if cooldown_left < GESTURE_COOLDOWN_S:
        bar_w = int(w * (1 - cooldown_left / GESTURE_COOLDOWN_S))
        cv2.rectangle(frame, (0, 54), (bar_w, 60), (0, 200, 90), -1)

    # Title
    cv2.putText(frame, "Gesture TV Controller  |  Q to quit",
                (10, 38), FONT, 0.65, (190, 190, 190), 1, cv2.LINE_AA)

    # Status
    if cooldown_left > 0:
        status, sc = f"cooldown {cooldown_left:.1f}s", (80, 150, 255)
    elif not index_extended:
        status, sc = "point index finger", (200, 180, 80)
    else:
        status, sc = "ready", (0, 220, 80)
    cv2.putText(frame, status, (w - 220, 38), FONT, 0.60, sc, 1, cv2.LINE_AA)

    # Big gesture label (centre frame, visible during cooldown)
    if last_gesture and (cooldown_left > 0.05 or tu_cooldown > 0.05):
        label = GESTURE_LABELS.get(last_gesture, last_gesture)
        col   = GESTURE_COLORS.get(last_gesture, (0, 230, 90))
        scale, thick = 2.0, 3
        (tw, th), _ = cv2.getTextSize(label, FONT, scale, thick)
        tx, ty = (w - tw) // 2, h // 2 + th // 2
        cv2.putText(frame, label, (tx + 2, ty + 2), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, ty), FONT, scale, col, thick, cv2.LINE_AA)

    # Thumbs-up indicator (bottom-left)
    bx, by = 10, h - 36
    tu_cooldown = max(0.0, THUMBS_UP_COOLDOWN_S - (time.time() - state.last_thumbsup_time))
    tu_ready = tu_cooldown == 0
    tu_col = (0, 220, 80) if tu_ready else (60, 60, 60)
    cv2.putText(frame, "Thumbs up = OK", (bx, by - 6), FONT, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
    label_tu = "READY" if tu_ready else f"cooldown {tu_cooldown:.1f}s"
    cv2.putText(frame, label_tu, (bx, by + 10), FONT, 0.48, tu_col, 1, cv2.LINE_AA)

    # Instruction hint (bottom-right)
    hints = ["Point index → swipe", "Thumb+index → OK"]
    for i, hint in enumerate(hints):
        cv2.putText(frame, hint, (w - 210, h - 36 + i * 20),
                    FONT, 0.45, (120, 120, 120), 1, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ensure_model()

    if ADB_HOST:
        import subprocess
        print(f"Connecting ADB to {ADB_HOST} …")
        r = subprocess.run(["adb", "connect", ADB_HOST], capture_output=True, text=True)
        print(r.stdout.strip())

    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("ERROR: Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state          = GestureState()
    last_gesture   = None
    start_time     = time.time()

    print("Running — point your index finger and swipe. Press Q to quit.")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)  # mirror so gestures feel natural
            h, w  = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms    = int((time.time() - start_time) * 1000)
            result   = landmarker.detect_for_video(mp_image, ts_ms)

            gesture        = None
            index_extended = False

            if result.hand_landmarks:
                lms       = result.hand_landmarks[0]
                index_tip = lms[8]   # INDEX_FINGER_TIP  ← swipe tracking point

                index_extended = is_index_extended(lms)
                draw_hand(frame, lms, index_extended)

                # Only accumulate swipe history when index is pointed out
                if index_extended:
                    state.push(index_tip.x, index_tip.y)
                else:
                    state.clear_history()

                # Draw swipe trail on index tip
                draw_swipe_trail(frame, state.history, h, w)

                # Thumbs-up fires independently of swipe cooldown
                if state.check_thumbs_up(lms):
                    gesture = "CENTER"
                elif index_extended:
                    gesture = state.check_swipe()

                if gesture:
                    last_gesture = gesture
                    print(f"[GESTURE] {gesture}")
                    if ADB_HOST:
                        send_adb_key(gesture)

            else:
                state.clear_history()

            cooldown_left = max(0.0, GESTURE_COOLDOWN_S - (time.time() - state.last_fire_time))
            tu_cooldown   = max(0.0, THUMBS_UP_COOLDOWN_S - (time.time() - state.last_thumbsup_time))
            draw_hud(frame, state, last_gesture, cooldown_left, tu_cooldown, index_extended)

            cv2.imshow("Gesture TV Controller", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Exited.")


if __name__ == "__main__":
    main()