from typing import List, Dict
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance

def calculate_area(coordinates: List[Dict[str, float]]) -> float:
    if len(coordinates) < 3:
        return 0.0

    ref_lat = coordinates[0]['lat']
    ref_lon = coordinates[0]['lon']

    points = []
    for coord in coordinates:
        dy = haversine_distance(ref_lat, ref_lon, coord['lat'], ref_lon)
        dx = haversine_distance(ref_lat, ref_lon, ref_lat, coord['lon'])

        if coord['lat'] < ref_lat:
            dy = -dy
        if coord['lon'] < ref_lon:
            dx = -dx

        points.append((dx, dy))

    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0

    return area

def calculate_polygon_centroid(coordinates: List[Dict[str, float]]) -> Dict[str, float]:
    if not coordinates:
        return None

    lat_sum = sum(coord['lat'] for coord in coordinates)
    lon_sum = sum(coord['lon'] for coord in coordinates)
    n = len(coordinates)

    return {
        'lat': lat_sum / n,
        'lon': lon_sum / n
    }