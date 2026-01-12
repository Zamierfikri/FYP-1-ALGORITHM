import pandas as pd
from anomaly_detection import detect_anomaly
from geofence import geofence_breach

df = pd.read_csv("ekf_output.csv")

def determine_alert(inside, anomaly):
    if inside and not anomaly:
        return "NO ALERT"
    elif inside and anomaly:
        return "SLIGHT ALERT"
    elif not inside and not anomaly:
        return "SLIGHT ALERT"
    else:
        return "HIGH ALERT"

results = []

for _, r in df.iterrows():
    inside = not geofence_breach(r["fused_lat"], r["fused_lon"])
    anomaly = detect_anomaly(r["ax"], r["ay"], r["az"], r["speed"])
    alert = determine_alert(inside, anomaly)

    results.append([
        r["time"],
        r["fused_lat"],
        r["fused_lon"],
        r["speed"],
        inside,
        anomaly,
        alert
    ])

out = pd.DataFrame(results, columns=[
    "time", "lat", "lon", "speed",
    "inside_geofence", "anomaly", "alert"
])

out.to_csv("final_alert_output.csv", index=False)
print("✅ Final alert generation completed → final_alert_output.csv")
