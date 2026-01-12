import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv("ekf_output.csv")

acc_mag = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
acc_delta = acc_mag.diff().fillna(0)
speed = df["speed"]

X = np.column_stack((acc_mag, acc_delta, speed))

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

model.fit(X)

joblib.dump(model, "anomaly_model.pkl")
print("✅ Anomaly model trained → anomaly_model.pkl")
