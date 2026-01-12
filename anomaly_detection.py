import numpy as np
import joblib

print("anomaly_detection.py loaded correctly")

model = joblib.load("anomaly_model.pkl")

_prev_acc = None

def detect_anomaly(ax, ay, az, speed):
    global _prev_acc

    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    if _prev_acc is None:
        acc_delta = 0.0
    else:
        acc_delta = abs(acc_mag - _prev_acc)

    _prev_acc = acc_mag

    features = [[acc_mag, acc_delta, speed]]

    return model.predict(features)[0] == -1
