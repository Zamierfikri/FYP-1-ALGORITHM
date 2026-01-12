import json

# ======================================================
# LOAD GEOJSON POLYGON
# ======================================================
with open("geofence_boundary.geojson", "r") as f:
    geojson = json.load(f)

# Extract polygon coordinates (lon, lat)
polygon = geojson["features"][0]["geometry"]["coordinates"][0]

# ======================================================
# POINT-IN-POLYGON (RAY CASTING ALGORITHM)
# ======================================================
def point_in_polygon(lat, lon, polygon):
    """
    lat, lon : EKF fused GPS position
    polygon  : list of (lon, lat) tuples
    """
    x = lon
    y = lat

    inside = False
    n = len(polygon)

    for i in range(n):
        lon1, lat1 = polygon[i]
        lon2, lat2 = polygon[(i + 1) % n]

        if ((lat1 > y) != (lat2 > y)):
            x_intersect = (lon2 - lon1) * (y - lat1) / (lat2 - lat1 + 1e-12) + lon1
            if x < x_intersect:
                inside = not inside

    return inside

# ======================================================
# GEOFENCE CHECK FUNCTION
# ======================================================
def geofence_breach(lat, lon):
    """
    Returns True if OUTSIDE geofence
    """
    return not point_in_polygon(lat, lon, polygon)
