#  Gesture TV Controller

Control your Android TV, Google TV, or any Android device hands-free using your webcam and hand gestures. No remote needed.

Built with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) for hand tracking, [OpenCV](https://opencv.org/) for the camera feed, and [ADB](https://developer.android.com/tools/adb) to send commands to your device over Wi-Fi.

---

##  Features

- **Real-time hand tracking** via webcam using Google's MediaPipe Hand Landmarker
- **4 swipe directions** — left, right, up, down — mapped to DPAD navigation
- **Thumbs up** gesture mapped to OK / Select
- **Anti-ghost-swipe logic** — returning your hand after a swipe never double-fires
- **Visual feedback** — big green gesture label appears on screen when detected
- **Live debug overlay** — swipe trail, finger state, cooldown bar all visible in the window
- **Works over Wi-Fi** — no USB cable required

---

## 🎮 Gestures

| Gesture | Action | How to do it |
|---|---|---|
| Swipe Left | DPAD Left | Point index finger, flick left |
| Swipe Right | DPAD Right | Point index finger, flick right |
| Swipe Up | DPAD Up | Point index finger, flick up |
| Swipe Down | DPAD Down | Point index finger, flick down |
| Thumbs Up | OK / Select | Raise thumb, curl all other fingers |
| Open Palm | Back | Extend all five fingers open toward the camera |

---

## How It Works

### Hand Tracking
The script uses MediaPipe's `HandLandmarker` task to detect 21 landmarks on your hand every frame. All processing runs locally on your machine — no cloud, no internet required after the model downloads.

### Swipe Detection
Only the **index fingertip** (landmark `#8`) is tracked for swipes — not the wrist. This makes detection much more precise and intentional.

For a swipe to register it must pass three checks:
- **Distance** — the fingertip must travel at least `0.12` normalised units across the frame
- **Velocity** — it must move fast enough to not be a slow drift
- **Directionality** — the movement must be at least 70% along one axis (so diagonal flicks don't fire)

After a swipe fires the position history is immediately cleared, so bringing your hand back can never retrigger it. A 1-second cooldown adds a second layer of protection.

### Index Finger Gate
Swipes only accumulate when your index finger is **actually extended**. This is detected by measuring the Euclidean distance between the fingertip and its base knuckle (MCP joint) — direction-agnostic, so pointing down works just as well as pointing up.

### Thumbs-Up Detection
Two conditions must both be true simultaneously:
1. Thumb tip is at least `0.08` normalised units **above** the thumb base knuckle
2. All four other fingers are **curled** — each fingertip is below its PIP (middle) joint

Fires immediately when both conditions are met, then waits 1 second before firing again.

### ADB Commands
When a gesture is recognised the script calls:
```
adb shell input keyevent <code>
```
over Wi-Fi to your device. Keycode mapping:

| Gesture | Keycode |
|---|---|
| Left | 21 |
| Right | 22 |
| Up | 19 |
| Down | 20 |
| OK | 23 |

---

## Installation

**Install dependencies:**
```bash
uv pip install mediapipe opencv-contrib-python
```

The MediaPipe hand landmark model (`hand_landmarker.task`, ~10 MB) downloads automatically on first run.

---

## Device Setup

### Android TV / Google TV

1. **Enable Developer Options**
   Settings → System → About → tap **Build Number** 7 times

2. **Enable Network Debugging**
   Settings → System → Developer Options → turn on **Network Debugging**

3. **Find your device's IP**
   Settings → Network & Internet → your Wi-Fi name → IP address shown at the bottom

4. **Connect from your PC**
   ```bash
   adb connect 192.168.x.x:5555
   ```
   Approve the popup on your TV screen.

### Android Phone (Android 11+)

1. **Enable Developer Options**
   Settings → About Phone → tap **Build Number** 7 times

2. **Enable Wireless Debugging**
   Settings → System → Developer Options → **Wireless Debugging** on

3. **Pair your PC** (one-time only)
   Tap **"Pair device with pairing code"** on your phone. It shows a pairing port and 6-digit code. Run:
   ```bash
   adb pair 192.168.x.x:PAIRING_PORT
   ```
   Enter the 6-digit code when prompted.

4. **Connect**
   Back on the main Wireless Debugging screen, use the port shown there:
   ```bash
   adb connect 192.168.x.x:CONNECTION_PORT
   ```

5. **Verify**
   ```bash
   adb devices
   ```
   Your device should show as `device` (not `unauthorized`).

> ⚠️ Your PC and device must be on the **same Wi-Fi network**. The connection port on phones changes every time Wireless Debugging is toggled — always check the screen for the current port.

---

## Running

1. Edit line 38 of `gesture_tv_controller.py`:
   ```python
   ADB_HOST = "192.168.x.x:5555"   # your device's IP and port
   ```

2. Run:
   ```bash
   python gesture_tv_controller.py
   ```

3. A window opens showing your webcam feed. Show your hand and start gesturing.

Press **Q** to quit.

---

## Tuning

All detection parameters are at the top of the script and easy to adjust:

| Parameter | Default | What it does |
|---|---|---|
| `MIN_SWIPE_DISTANCE` | `0.12` | How far the finger must travel — increase if accidental swipes fire |
| `MIN_SWIPE_VELOCITY` | `0.35` | Minimum speed — increase to require faster, more deliberate swipes |
| `DIRECTIONALITY_RATIO` | `0.70` | How straight the swipe must be — increase to reject diagonal flicks |
| `GESTURE_COOLDOWN_S` | `1.0` | Seconds before another swipe can fire |
| `INDEX_EXTEND_THRESHOLD` | `0.14` | How extended the index finger must be |
| `THUMB_RISE_MIN` | `0.08` | How high the thumb must be raised for thumbs-up |
| `THUMBS_UP_COOLDOWN_S` | `1.0` | Seconds before thumbs-up can fire again |

