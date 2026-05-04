# Personalized Temporal Driver Drowsiness Detection System (Industrial Roadmap)

## Project Vision
Build a **real-time driver monitoring system** that detects drowsiness using:
- Eye movement (EAR)
- Yawning detection (MAR)
- Head movement tracking
- Temporal modeling using LSTM
- Personalized baseline adaptation
- Low-light enhancement
- Severity prediction (Low / Medium / High)
- Real-time alert mechanism

---

# 1. Problem Statement
Traditional systems:
- Use fixed thresholds
- Binary classification (drowsy/not drowsy)
- Fail in low light
- Ignore driver-specific facial differences
- Ignore temporal behavior

### Our Solution
A real-time adaptive system that:
1. Learns driver baseline behavior
2. Tracks facial patterns over time
3. Predicts severity levels
4. Works in low-light conditions
5. Generates real-time alerts

---

# 2. Novelty Statement
Use this when your guide asks:

**"Existing systems rely on fixed thresholds and frame-level detection. Our proposed system introduces personalized baseline learning, temporal LSTM-based modeling, severity classification, low-light enhancement, and multi-modal feature fusion for real-time adaptive drowsiness detection."**

---

# 3. Full Architecture

```text
Camera Input
→ Low Light Enhancement
→ Face Detection
→ Facial Landmark Extraction
→ Feature Extraction
→ Personalization Layer
→ Temporal LSTM Model
→ Severity Prediction
→ Alert Engine
→ Dashboard + Logging
```

---

# 4. Tech Stack

## Computer Vision
- OpenCV
- MediaPipe Face Mesh
- Dlib (optional)

## Deep Learning
- TensorFlow / Keras
OR
- PyTorch

## Deployment
- Python
- Flask/FastAPI (optional)
- Streamlit dashboard (optional)

## Alerts
- Pygame
OR
- winsound

## Database/Storage
- CSV
- SQLite

---

# 5. Phase 1: Face Detection

## Goal
Detect face in real-time.

## Tasks
- Webcam integration
- Face bounding box
- Face tracking

## Output
Live webcam feed with detected face.

---

# 6. Phase 2: Landmark Detection

Extract:
- Eye landmarks
- Mouth landmarks
- Nose landmarks

## Output
Display facial landmarks.

---

# 7. Phase 3: Feature Engineering

## Eye Feature
EAR (Eye Aspect Ratio)

## Mouth Feature
MAR (Mouth Aspect Ratio)

## Head Pose Features
- Pitch
- Roll
- Yaw

## Additional Features
- Blink frequency
- Eye closure duration
- Yawn frequency

### Feature Vector
```text
[EAR, MAR, Pitch, Roll, Yaw, Blink Count, Yawn Count]
```

---

# 8. Phase 4: Baseline Rule-Based System

Build non-AI system first.

Rules:
- EAR < threshold
- MAR > threshold
- Head tilt > threshold

Output:
- Drowsy
- Not Drowsy

This becomes baseline comparison model.

---

# 9. Phase 5: Personalization Layer (Novelty)

## Problem
Different users have different normal EAR values.

Example:
- User A → 0.30
- User B → 0.25

## Solution
During first 60 seconds:
- Collect normal driving behavior
- Learn personalized baseline

Formula:

```text
Normalized EAR = Current EAR / Personal EAR Baseline
```

Same applies to:
- MAR
- Head movement

---

# 10. Phase 6: Dataset Collection

Collect your own dataset.

## Record scenarios
- Alert state
- Mild drowsiness
- Moderate drowsiness
- High drowsiness
- Low-light scenarios

## Save
Frame-wise feature logs.

Example:

```text
EAR, MAR, Pitch, Label
```

---

# 11. Phase 7: LSTM Temporal Model

## Why LSTM?
Drowsiness happens over time.

Input:
30–60 frame sequences

```text
[t1,t2,t3,t4....tn]
```

Model:

```text
Input
→ LSTM
→ Dense
→ Softmax
```

Output Classes:
- Alert
- Low
- Medium
- High

---

# 12. Phase 8: Severity Classification

Instead of binary output.

| Level | Meaning |
|--------|----------|
| Alert | Normal |
| Low | Mild fatigue |
| Medium | Drowsy |
| High | Dangerous |

---

# 13. Phase 9: Low Light Enhancement

Apply before detection:

- CLAHE
- Gamma correction
- Histogram equalization

Goal:
Improve night-time detection.

---

# 14. Phase 10: Alert Engine

## Alert Logic

Low:
- Warning text

Medium:
- Beep alert

High:
- Continuous alarm

Optional:
- SMS alert
- Seat vibration simulation

---

# 15. Phase 11: Dashboard

Display:
- EAR
- MAR
- Head angle
- Severity score
- Alert status
- FPS

---

# 16. Phase 12: Evaluation Metrics

Compare:

## Without LSTM
Traditional ML/threshold

## With LSTM
Temporal model

Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- FPS
- Latency

---

# 17. Folder Structure

```text
project/
 ├── dataset/
 ├── models/
 ├── logs/
 ├── alerts/
 ├── app.py
 ├── feature_extraction.py
 ├── train_lstm.py
 ├── personalize.py
 ├── low_light.py
 └── dashboard.py
```

---

# 18. Timeline

## Week 1
Face + landmarks

## Week 2
Feature extraction

## Week 3
Personalization

## Week 4
Dataset collection

## Week 5
LSTM training

## Week 6
Alerts + dashboard

## Week 7
Evaluation

## Week 8
Final testing

---

# 19. Future Scope

- IR camera support
- Mobile deployment
- Edge AI deployment
- Fleet monitoring dashboard
- Driver health integration

---

# Final Pitch

**"This project transforms traditional drowsiness detection from static threshold systems into an adaptive real-time personalized AI safety platform."**

