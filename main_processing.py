import pandas as pd
from geofence import geofence_breach
from anomaly_detection import detect_anomaly

print("✅ main_processing.py STARTED")
# ======================================================
# LOAD EKF OUTPUT
# ======================================================
df = pd.read_csv("ekf_output.csv")

results = []

# ======================================================
# ALERT LOGIC FUNCTION
# ======================================================
def determine_alert(inside_geofence, anomaly):
    if inside_geofence and not anomaly:
        return "NO ALERT"
    elif inside_geofence and anomaly:
        return "SLIGHT ALERT"
    elif not inside_geofence and not anomaly:
        return "SLIGHT ALERT"
    else:
        return "HIGH ALERT"

# ======================================================
# MAIN PROCESSING LOOP
# ======================================================
for _, r in df.iterrows():

    # --- Geofence check (using EKF fused GPS) ---
    inside_geofence = not geofence_breach(
        r["fused_lat"],
        r["fused_lon"]
    )

    # --- Anomaly detection (using EKF + IMU features) ---
    anomaly = detect_anomaly(
        r["ax"],
        r["ay"],
        r["az"],
        r["speed"]
    )

    # --- Alert decision ---
    alert = determine_alert(inside_geofence, anomaly)

    results.append([
        r["time"],
        r["fused_lat"],
        r["fused_lon"],
        r["speed"],
        inside_geofence,
        anomaly,
        alert
    ])

# ======================================================
# SAVE FINAL OUTPUT
# ======================================================
out_df = pd.DataFrame(
    results,
    columns=[
        "time",
        "lat",
        "lon",
        "speed",
        "inside_geofence",
        "anomaly",
        "alert"
    ]
)

out_df.to_csv("final_alert_output.csv", index=False)

print("✅ Main processing completed successfully")
print("📄 Output saved as final_alert_output.csv")
