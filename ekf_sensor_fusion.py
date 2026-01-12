import pandas as pd
import numpy as np
import math

# ======================================================
# LOAD DATASETS (ENCODING FIXED)
# ======================================================
imu = pd.read_csv("Accel.csv", encoding="latin1")
gps = pd.read_csv("gps_data.csv", encoding="latin1")

# ======================================================
# CLEAN COLUMN NAMES (IMPORTANT FIX)
# ======================================================
imu.columns = imu.columns.str.strip().str.lower()
gps.columns = gps.columns.str.strip().str.lower()

# ======================================================
# RENAME IMU COLUMNS (ROBUST)
# ======================================================
imu.rename(columns={
    "time (s)": "time",
    "acceleration x (m/s^2)": "ax",
    "acceleration y (m/s^2)": "ay",
    "acceleration z (m/s^2)": "az",
    "acceleration x (m/s²)": "ax",
    "acceleration y (m/s²)": "ay",
    "acceleration z (m/s²)": "az"
}, inplace=True)

# ======================================================
# RENAME GPS COLUMNS
# ======================================================
gps.rename(columns={
    "time (s)": "time",
    "latitude (deg)": "lat",
    "longitude (deg)": "lon"
}, inplace=True)

# ======================================================
# VERIFY REQUIRED COLUMNS EXIST
# ======================================================
required_imu = {"time", "ax", "ay", "az"}
required_gps = {"time", "lat", "lon"}

if not required_imu.issubset(imu.columns):
    raise ValueError(f"IMU missing columns: {required_imu - set(imu.columns)}")

if not required_gps.issubset(gps.columns):
    raise ValueError(f"GPS missing columns: {required_gps - set(gps.columns)}")

# ======================================================
# TIME SYNCHRONIZATION
# ======================================================
data = pd.merge_asof(
    imu.sort_values("time"),
    gps.sort_values("time"),
    on="time",
    direction="nearest"
)

# ======================================================
# GPS → LOCAL CARTESIAN
# ======================================================
LAT0 = data.loc[0, "lat"]
LON0 = data.loc[0, "lon"]

def gps_to_xy(lat, lon):
    x = (lon - LON0) * 111000 * math.cos(math.radians(LAT0))
    y = (lat - LAT0) * 111000
    return x, y

def xy_to_gps(x, y):
    lat = LAT0 + y / 111000
    lon = LON0 + x / (111000 * math.cos(math.radians(LAT0)))
    return lat, lon

# ======================================================
# EKF INITIALIZATION
# ======================================================
dt = data["time"].diff().mean()

x = np.zeros((4, 1))   # [x, y, vx, vy]
P = np.eye(4) * 10

Q = np.diag([0.5, 0.5, 1.0, 1.0])
R = np.diag([25, 25])

H = np.array([[1,0,0,0],[0,1,0,0]])

# ======================================================
# EKF LOOP
# ======================================================
fused_data = []

for _, r in data.iterrows():
    ax, ay = r["ax"], r["ay"]

    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    B = np.array([[0.5*dt**2,0],[0,0.5*dt**2],[dt,0],[0,dt]])
    u = np.array([[ax],[ay]])

    x = F @ x + B @ u
    P = F @ P @ F.T + Q

    gx, gy = gps_to_xy(r["lat"], r["lon"])
    z = np.array([[gx],[gy]])

    yk = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x = x + K @ yk
    P = (np.eye(4) - K @ H) @ P

    px, py, vx, vy = x.flatten()
    speed = np.sqrt(vx**2 + vy**2)
    fused_lat, fused_lon = xy_to_gps(px, py)

    fused_data.append([
        r["time"], fused_lat, fused_lon, speed,
        r["ax"], r["ay"], r["az"]
    ])

# ======================================================
# SAVE OUTPUT
# ======================================================
output_df = pd.DataFrame(
    fused_data,
    columns=["time","fused_lat","fused_lon","speed","ax","ay","az"]
)

output_df.to_csv("ekf_output.csv", index=False)
print("✅ EKF fusion completed successfully")
