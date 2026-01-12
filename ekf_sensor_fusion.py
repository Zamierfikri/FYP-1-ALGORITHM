import pandas as pd
import numpy as np

# ======================================================
# CONSTANTS
# ======================================================
EARTH_RADIUS = 6371000  # meters
DT = 1.0               # sampling time (s)

# ======================================================
# LOAD DATA
# ======================================================
imu = pd.read_csv("Accel.csv", encoding="latin1")
gps = pd.read_csv("gps_data.csv", encoding="latin1")

# ======================================================
# CLEAN COLUMN NAMES
# ======================================================
imu.columns = imu.columns.str.strip().str.lower()
gps.columns = gps.columns.str.strip().str.lower()

imu.rename(columns={
    "time (s)": "time",
    "acceleration x (m/s^2)": "ax",
    "acceleration y (m/s^2)": "ay",
    "acceleration z (m/s^2)": "az"
}, inplace=True)

gps.rename(columns={
    "time (s)": "time",
    "latitude (deg)": "lat",
    "longitude (deg)": "lon"
}, inplace=True)

print("✅ Columns mapped correctly")

# ======================================================
# GPS → LOCAL METERS (ENU)
# ======================================================
lat0 = gps.loc[0, "lat"]
lon0 = gps.loc[0, "lon"]

def latlon_to_xy(lat, lon):
    x = np.deg2rad(lon - lon0) * EARTH_RADIUS * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * EARTH_RADIUS
    return x, y

def xy_to_latlon(x, y):
    lat = np.rad2deg(y / EARTH_RADIUS) + lat0
    lon = np.rad2deg(x / (EARTH_RADIUS * np.cos(np.deg2rad(lat0)))) + lon0
    return lat, lon

# ======================================================
# EKF INITIALIZATION
# State: [x, y, vx, vy] in meters & m/s
# ======================================================
x = np.zeros((4, 1))
P = np.eye(4)

# ---------------- METHOD 1 TUNING ----------------
# Reduce IMU influence
Q = np.diag([0.01, 0.01, 0.05, 0.05])

# Trust GPS more
R = np.diag([0.5, 0.5])

H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

out = []

N = min(len(imu), len(gps))

# ======================================================
# EKF LOOP
# ======================================================
for i in range(N):

    # Read IMU
    ax = imu.loc[i, "ax"]
    ay = imu.loc[i, "ay"]

    # ---------------- IMPORTANT: CLAMP ACCELERATION ----------------
    ax = np.clip(ax, -2.0, 2.0)
    ay = np.clip(ay, -2.0, 2.0)

    # ---------------- PREDICTION ----------------
    F = np.array([
        [1, 0, DT, 0],
        [0, 1, 0, DT],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    B = np.array([
        [0.5 * DT**2, 0],
        [0, 0.5 * DT**2],
        [DT, 0],
        [0, DT]
    ])

    u = np.array([[ax], [ay]])

    x = F @ x + B @ u
    P = F @ P @ F.T + Q

    # ---------------- UPDATE (GPS) ----------------
    gx, gy = latlon_to_xy(gps.loc[i, "lat"], gps.loc[i, "lon"])
    z = np.array([[gx], [gy]])

    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x = x + K @ y
    P = (np.eye(4) - K @ H) @ P

    # ---------------- OUTPUT ----------------
    lat, lon = xy_to_latlon(x[0, 0], x[1, 0])
    speed = np.sqrt(x[2, 0]**2 + x[3, 0]**2)

    out.append([
        gps.loc[i, "time"],
        lat,
        lon,
        speed,
        imu.loc[i, "ax"],
        imu.loc[i, "ay"],
        imu.loc[i, "az"]
    ])

# ======================================================
# SAVE OUTPUT
# ======================================================
df = pd.DataFrame(out, columns=[
    "time",
    "fused_lat",
    "fused_lon",
    "speed",
    "ax",
    "ay",
    "az"
])

df.to_csv("ekf_output.csv", index=False)

print("✅ EKF (Method 1) completed successfully")
print("📄 Output saved as ekf_output.csv")
