# FYP-1 Algorithm Suite — Sensor Fusion for Crowd Geofencing Anomaly Detection

> **Final Year Project 1 (FYP-1)**
> Sensor Fusion for Crowd Geofencing Anomaly Detection
> Language: Python 3

---

## Overview

This repository contains the core algorithm pipeline developed for **Sensor Fusion for Crowd Geofencing Anomaly Detection**. The system fuses GPS and IMU sensor data using an Extended Kalman Filter (EKF), applies unsupervised machine learning to detect abnormal motion patterns, enforces geofence boundaries using spatial analysis, and generates tiered alert levels for real-time situational awareness.

The pipeline is structured as five modular Python scripts that work in sequence — from raw sensor ingestion through to final alert classification output.

---

## System Architecture

```
Accel.csv (IMU)  ──┐
                   ├──► ekf_sensor_fusion.py ──► ekf_output.csv
gps_data.csv (GPS) ┘                                    │
                                                         ├──► train_anomaly_detection.py ──► anomaly_model.pkl
                                                         │
geofence_boundary.geojson ──► geofence.py               │
anomaly_model.pkl ──────────► anomaly_detection.py       │
                                                         │
                              main_processing.py ◄───────┘
                                      │
                                      ▼
                              final_alert_output.csv
```

---

## Modules

### 1. `ekf_sensor_fusion.py` — Extended Kalman Filter (EKF)

Fuses raw GPS and IMU (accelerometer) data into a single, noise-reduced position and velocity estimate using an Extended Kalman Filter.

**Inputs:**
- `Accel.csv` — IMU accelerometer data (`time`, `ax`, `ay`, `az` in m/s²)
- `gps_data.csv` — GPS coordinates (`time`, `lat`, `lon` in degrees)

**Process:**
- Converts GPS coordinates to local ENU (East-North-Up) metric frame
- Runs EKF with state vector `[x, y, vx, vy]`
- Acceleration is clamped to ±2.0 m/s² to suppress IMU noise
- Process noise `Q` is tuned to reduce IMU drift; measurement noise `R` is tuned to trust GPS

**Output:**
- `ekf_output.csv` — Fused position (`fused_lat`, `fused_lon`), speed, and raw IMU values

---

### 2. `train_anomaly_detection.py` — Isolation Forest Training

Trains an unsupervised machine learning model to learn normal crowd motion patterns from EKF-processed data.

**Input:**
- `ekf_output.csv` — Output from EKF fusion

**Features used:**
- `acc_mag` — Total acceleration magnitude (√(ax² + ay² + az²))
- `acc_delta` — Change in acceleration magnitude between timesteps
- `speed` — Fused speed from EKF

**Model:**
- Algorithm: `IsolationForest` (scikit-learn)
- Estimators: 100 trees
- Contamination: 5% (assumed anomaly rate)
- Random state: 42 (reproducible)

**Output:**
- `anomaly_model.pkl` — Serialised trained model (via `joblib`)

> **Note:** Run this script once before running `main_processing.py` to generate the model file.

---

### 3. `anomaly_detection.py` — Real-Time Anomaly Detection

Loads the trained Isolation Forest model and provides a callable function for per-timestep anomaly classification.

**Input:**
- `anomaly_model.pkl` — Pre-trained model
- Per-row sensor values: `ax`, `ay`, `az`, `speed`

**Logic:**
- Computes `acc_mag` and `acc_delta` from live IMU values
- Returns `True` if the model predicts an anomaly (`prediction == -1`)

---

### 4. `geofence.py` — Geofence Boundary Check

Implements a point-in-polygon algorithm to determine whether a tracked entity (person or device) is operating within the defined geofence boundary — the crowd monitoring zone.

**Input:**
- `geofence_boundary.geojson` — A GeoJSON file defining the allowed operational polygon

**Algorithm:**
- Ray casting method for robust point-in-polygon testing
- Returns `True` (breach detected) if the fused GPS position is **outside** the defined crowd geofence zone

---

### 5. `main_processing.py` — Alert Generation Pipeline

Integrates all modules to produce the final alert classification for each timestep.

**Input:**
- `ekf_output.csv`

**Alert Logic:**

| Inside Geofence | Anomaly Detected | Alert Level  |
|:--------------:|:----------------:|:------------:|
| ✅ Yes          | ❌ No            | NO ALERT     |
| ✅ Yes          | ✅ Yes           | SLIGHT ALERT |
| ❌ No           | ❌ No            | SLIGHT ALERT |
| ❌ No           | ✅ Yes           | HIGH ALERT   |

**Output:**
- `final_alert_output.csv` — Columns: `time`, `lat`, `lon`, `speed`, `inside_geofence`, `anomaly`, `alert`

---

## Required Data Files

The following data files are **not included** in this repository and must be prepared separately:

| File | Description |
|------|-------------|
| `Accel.csv` | IMU accelerometer data from the tracked entity/device |
| `gps_data.csv` | GPS coordinates logged during data collection |
| `geofence_boundary.geojson` | GeoJSON polygon defining the crowd monitoring zone |

---

## Dependencies

Install required Python packages via pip:

```bash
pip install pandas numpy scikit-learn joblib
```

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and CSV I/O |
| `numpy` | Numerical computation (EKF matrices, vector math) |
| `scikit-learn` | Isolation Forest anomaly detection |
| `joblib` | Model serialisation |

---

## Usage

Run the scripts in the following order:

```bash
# Step 1: Fuse GPS + IMU data
python ekf_sensor_fusion.py

# Step 2: Train the anomaly detection model
python train_anomaly_detection.py

# Step 3: Run the full alert pipeline
python main_processing.py
```

The final output will be saved as `final_alert_output.csv`.

---

## Project Context

This algorithm suite was developed as part of **FYP-1** with the research topic **"Sensor Fusion for Crowd Geofencing Anomaly Detection"**. The project combines Extended Kalman Filter-based sensor fusion (GPS + IMU) with an Isolation Forest machine learning model to detect anomalous crowd behaviour and geofence violations in real time. The system is designed to improve situational awareness in crowd monitoring scenarios by providing automated, tiered alert outputs.

---

## Author

**Zamier Fikri**
Mechatronics Engineering Student
[GitHub Profile](https://github.com/Zamierfikri)
