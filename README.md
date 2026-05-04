# Personalized Temporal Driver Drowsiness Detection System

Real-time driver monitoring system for drowsiness detection using eye movement, yawning, head pose, personalized baselines, low-light enhancement, and optional temporal LSTM severity classification.

## Features

- Real-time webcam monitoring with OpenCV
- Low-light enhancement using gamma correction, CLAHE, and histogram equalization
- Face landmark detection using MediaPipe Face Landmarker
- EAR and MAR based eye and mouth analysis
- Head pose estimation with pitch, roll, and yaw
- Personalized baseline collection for the first 60 seconds
- Rule-based severity estimation for immediate inference
- Optional PyTorch LSTM temporal model for sequence-based severity classification
- Real-time alert engine
- Streamlit dashboard for reviewing logged data
- CSV logging for dataset creation and later model training

## Repository layout

```text
driveai/
├── alerts/
├── dataset/
├── logs/
├── models/
├── alert_engine.py
├── app.py
├── dashboard.py
├── dashboard_streamlit.py
├── feature_extraction.py
├── low_light.py
├── personalize.py
├── requirements.txt
├── start_all.bat
├── start_dashboard.bat
├── start_detector.bat
├── stop_all.bat
├── temporal_model.py
└── train_lstm.py
```

## One-tap run on Windows

After cloning the repository, the easiest path is:

1. Double-click `start_all.bat`
2. Wait for dependency installation on the first run
3. Use the desktop webcam window for live detection
4. Open the dashboard at [http://127.0.0.1:8501](http://127.0.0.1:8501)

This script will:

- create `.venv` if it does not exist
- install `requirements.txt`
- create required folders
- start the live detector
- start the Streamlit dashboard

To stop everything, double-click `stop_all.bat`.

## Manual setup

### 1. Clone the repository

```powershell
git clone <your-repo-url>
cd driveai
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Start the live detector

```powershell
python app.py
```

### 6. Start the dashboard in a separate terminal

```powershell
streamlit run dashboard_streamlit.py --server.address 127.0.0.1 --server.port 8501
```

## How the system works

```text
Webcam
-> Low Light Enhancement
-> Face Landmark Detection
-> Feature Extraction
-> Personalized Baseline Normalization
-> Rule-Based Severity / Optional LSTM
-> Alert Engine
-> CSV Logging
-> Streamlit Dashboard
```

## Main components

### `app.py`

Runs the end-to-end real-time detection loop.

Responsibilities:

- opens webcam stream
- enhances low-light frames
- extracts facial features
- normalizes features against personal baseline
- predicts severity
- triggers alerts
- logs frame-level signals
- renders the on-screen dashboard

### `feature_extraction.py`

Handles:

- MediaPipe face landmark model download
- face landmark inference
- eye aspect ratio
- mouth aspect ratio
- pose estimation
- blink and yawn event counting

Note:

The MediaPipe face landmark model is downloaded automatically into `models/face_landmarker.task` on first run.

### `personalize.py`

Collects baseline values for:

- `ear`
- `mar`
- `pitch`
- `roll`
- `yaw`

After warm-up, normalized values such as `norm_ear` and `norm_mar` are used for better person-specific decisions.

### `alert_engine.py`

Maps features to severity levels:

- `alert`
- `low`
- `medium`
- `high`

Also plays Windows beeps for medium and high risk states.

### `dashboard_streamlit.py`

Reads `logs/session_features.csv` and shows:

- latest severity
- EAR and MAR trends
- pose trends
- blink, yawn, and eye-closure trends
- recent recorded rows

### `train_lstm.py`

Trains a temporal PyTorch LSTM on collected CSV data.

Outputs:

- validation accuracy
- weighted F1
- confusion matrix
- saved model in `models/drowsiness_lstm.pt`

## Labels and keyboard controls

While the OpenCV detector window is focused:

- `0` or `a` sets current label to `alert`
- `1` or `l` sets current label to `low`
- `2` or `m` sets current label to `medium`
- `3` or `h` sets current label to `high`
- `q` closes the detector

The active label is written into the CSV log so you can build a supervised dataset.

## Output files

### Runtime logs

- `logs/session_features.csv`
- `logs/app_stdout.log`
- `logs/app_stderr.log`
- `logs/dashboard_stdout.log`
- `logs/dashboard_stderr.log`

### Model files

- `models/face_landmarker.task`
- `models/drowsiness_lstm.pt`

## Training your own temporal model

1. Run the detector and collect samples for multiple states.
2. Use the label keys while recording.
3. Stop the app.
4. Train the sequence model:

```powershell
python train_lstm.py --csv logs/session_features.csv
```

5. The trained model will be saved to:

```text
models/drowsiness_lstm.pt
```

6. Restart the detector. If the model file exists, the app will use it after baseline warm-up.

## Recommended data collection protocol

Record separate sessions for:

- normal alert driving posture
- mild fatigue
- moderate drowsiness
- strong drowsiness
- low-light conditions

Try to capture:

- different head angles
- glasses vs no glasses
- different room brightness
- different distances from the camera

## Tested workflow

Verified locally:

- Python imports succeed
- project modules import cleanly
- Streamlit dashboard launches at `127.0.0.1:8501`
- detector process starts successfully
- LSTM training pipeline runs end-to-end with a smoke dataset

## Troubleshooting

### Webcam window does not appear

- Check that another app is not already using the webcam
- Re-run `start_detector.bat`
- Inspect `logs/app_stderr.log`

### Dashboard does not open

- Visit [http://127.0.0.1:8501](http://127.0.0.1:8501)
- Re-run `start_dashboard.bat`
- Inspect `logs/dashboard_stderr.log`

### First launch is slow

This is normal. The first run may:

- create a virtual environment
- install Python packages
- download the MediaPipe face landmark model

### Python 3.13 issues on another machine

If another machine struggles with package compatibility, use Python 3.11 or 3.12 for the smoothest setup.

# Personalized Temporal Driver Drowsiness Detection System

Real-time driver monitoring system for drowsiness detection using eye movement, yawning, head pose, personalized baselines, low-light enhancement, and optional temporal LSTM severity classification.

## Features

- Real-time webcam monitoring with OpenCV
- Low-light enhancement using gamma correction, CLAHE, and histogram equalization
- Face landmark detection using MediaPipe Face Landmarker
- EAR and MAR based eye and mouth analysis
- Head pose estimation with pitch, roll, and yaw
- Personalized baseline collection for the first 60 seconds
- Rule-based severity estimation for immediate inference
- Optional PyTorch LSTM temporal model for sequence-based severity classification
- Real-time alert engine
- Streamlit dashboard for reviewing logged data
- CSV logging for dataset creation and later model training

## Repository layout

```text
driveai/
├── alerts/
├── dataset/
├── logs/
├── models/
├── alert_engine.py
├── app.py
├── dashboard.py
├── dashboard_streamlit.py
├── feature_extraction.py
├── low_light.py
├── personalize.py
├── requirements.txt
├── start_all.bat
├── start_dashboard.bat
├── start_detector.bat
├── stop_all.bat
├── temporal_model.py
└── train_lstm.py
```

## One-tap run on Windows

After cloning the repository, the easiest path is:

1. Double-click `start_all.bat`
2. Wait for dependency installation on the first run
3. Use the desktop webcam window for live detection
4. Open the dashboard at [http://127.0.0.1:8501](http://127.0.0.1:8501)

This script will:

- create `.venv` if it does not exist
- install `requirements.txt`
- create required folders
- start the live detector
- start the Streamlit dashboard

To stop everything, double-click `stop_all.bat`.

## Manual setup

### 1. Clone the repository

```powershell
git clone <your-repo-url>
cd driveai
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Start the live detector

```powershell
python app.py
```

### 6. Start the dashboard in a separate terminal

```powershell
streamlit run dashboard_streamlit.py --server.address 127.0.0.1 --server.port 8501
```

## How the system works

```text
Webcam
-> Low Light Enhancement
-> Face Landmark Detection
-> Feature Extraction
-> Personalized Baseline Normalization
-> Rule-Based Severity / Optional LSTM
-> Alert Engine
-> CSV Logging
-> Streamlit Dashboard
```

## Main components

### `app.py`

Runs the end-to-end real-time detection loop.

Responsibilities:

- opens webcam stream
- enhances low-light frames
- extracts facial features
- normalizes features against personal baseline
- predicts severity
- triggers alerts
- logs frame-level signals
- renders the on-screen dashboard

### `feature_extraction.py`

Handles:

- MediaPipe face landmark model download
- face landmark inference
- eye aspect ratio
- mouth aspect ratio
- pose estimation
- blink and yawn event counting

Note:

The MediaPipe face landmark model is downloaded automatically into `models/face_landmarker.task` on first run.

### `personalize.py`

Collects baseline values for:

- `ear`
- `mar`
- `pitch`
- `roll`
- `yaw`

After warm-up, normalized values such as `norm_ear` and `norm_mar` are used for better person-specific decisions.

### `alert_engine.py`

Maps features to severity levels:

- `alert`
- `low`
- `medium`
- `high`

Also plays Windows beeps for medium and high risk states.

### `dashboard_streamlit.py`

Reads `logs/session_features.csv` and shows:

- latest severity
- EAR and MAR trends
- pose trends
- blink, yawn, and eye-closure trends
- recent recorded rows

### `train_lstm.py`

Trains a temporal PyTorch LSTM on collected CSV data.

Outputs:

- validation accuracy
- weighted F1
- confusion matrix
- saved model in `models/drowsiness_lstm.pt`

## Labels and keyboard controls

While the OpenCV detector window is focused:

- `0` or `a` sets current label to `alert`
- `1` or `l` sets current label to `low`
- `2` or `m` sets current label to `medium`
- `3` or `h` sets current label to `high`
- `q` closes the detector

The active label is written into the CSV log so you can build a supervised dataset.

## Output files

### Runtime logs

- `logs/session_features.csv`
- `logs/app_stdout.log`
- `logs/app_stderr.log`
- `logs/dashboard_stdout.log`
- `logs/dashboard_stderr.log`

### Model files

- `models/face_landmarker.task`
- `models/drowsiness_lstm.pt`

## Training your own temporal model

1. Run the detector and collect samples for multiple states.
2. Use the label keys while recording.
3. Stop the app.
4. Train the sequence model:

```powershell
python train_lstm.py --csv logs/session_features.csv
```

5. The trained model will be saved to:

```text
models/drowsiness_lstm.pt
```

6. Restart the detector. If the model file exists, the app will use it after baseline warm-up.

## Recommended data collection protocol

Record separate sessions for:

- normal alert driving posture
- mild fatigue
- moderate drowsiness
- strong drowsiness
- low-light conditions

Try to capture:

- different head angles
- glasses vs no glasses
- different room brightness
- different distances from the camera

## Tested workflow

Verified locally:

- Python imports succeed
- project modules import cleanly
- Streamlit dashboard launches at `127.0.0.1:8501`
- detector process starts successfully
- LSTM training pipeline runs end-to-end with a smoke dataset

## Troubleshooting

### Webcam window does not appear

- Check that another app is not already using the webcam
- Re-run `start_detector.bat`
- Inspect `logs/app_stderr.log`

### Dashboard does not open

- Visit [http://127.0.0.1:8501](http://127.0.0.1:8501)
- Re-run `start_dashboard.bat`
- Inspect `logs/dashboard_stderr.log`

### First launch is slow

This is normal. The first run may:

- create a virtual environment
- install Python packages
- download the MediaPipe face landmark model

### Python 3.13 issues on another machine

If another machine struggles with package compatibility, use Python 3.11 or 3.12 for the smoothest setup.
