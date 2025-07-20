# 🕹️ Hand Gesture Controlled Racing Game using MediaPipe

Control a car racing game using just your hand gestures and a webcam – no keyboard or joystick needed!

This project uses **MediaPipe Hands** and JavaScript to detect hand gestures in real-time and map them to game control keys like `ArrowUp` (accelerate), `ArrowDown` (brake), `ArrowLeft` (left), and `ArrowRight` (right). Works seamlessly with browser-based games like [Racing Limits](https://www.crazygames.com/game/racing-limits).

---

## 🚀 Features

- 🎮 Control car racing games with hand gestures
- ✋ Open palm → Accelerate (`↑`)
- ✊ Fist → Brake (`↓`)
- 👈 Left swipe → Steer left (`←`)
- 👉 Right swipe → Steer right (`→`)
- 📷 Real-time gesture detection with webcam
- 🔄 Optional Python version using `pyautogui` for full system control

---

## 🧠 Tech Stack

| Technology     | Purpose                          |
|----------------|----------------------------------|
| [MediaPipe](https://google.github.io/mediapipe/) | Hand landmark detection |
| JavaScript     | Gesture mapping & key simulation |
| HTML/CSS       | UI and overlays                  |
| OpenCV (Python) | Webcam capture (Python version) |
| pyautogui (Python) | Simulate keypress/mouse control (Python) |

---

## 🧑‍💻 Installation & Setup

### 📦 Web Version (Browser)

```bash
git clone https://github.com/your-username/hand-gesture-game-control.git](https://github.com/Uday-132/gesture_control/tree/main
python mediapipe_gesture_control.py
