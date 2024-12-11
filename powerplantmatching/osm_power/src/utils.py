import pandas as pd
from datetime import datetime
import re
from typing import Dict, List
from math import radians, sin, cos, sqrt, atan2

def parse_flexible_date(date_str):
    """
    Flexibly parse various date string formats into a datetime object.

    Args:
        date_str (str): Input date string to parse

    Returns:
        datetime: Parsed datetime object or None if parsing fails
    """
    if pd.isna(date_str):
        return None

    # Convert to string to handle potential non-string inputs
    date_str = str(date_str).strip()

    # Patterns to try in order
    date_patterns = [
        # ISO-like formats
        ('%Y-%m-%d', r'^\d{4}-\d{2}-\d{2}$'),
        ('%Y-%m', r'^\d{4}-\d{2}$'),

        # Full month name formats
        ('%B %Y', r'^[A-Za-z]+ \d{4}$'),

        # DD/MM/YYYY format
        ('%d/%m/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),
        ('%d-%m-%Y', r'^\d{1,2}-\d{1,2}-\d{4}$'),

        # Numeric year-only
        ('%Y', r'^\d{4}$'),
    ]

    for fmt, pattern in date_patterns:
        if re.match(pattern, date_str):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

    # Fallback for years
    try:
        # If it looks like a year, use January 1st
        if len(date_str) == 4 and date_str.isdigit():
            return datetime(int(date_str), 1, 1)
    except ValueError:
        pass
    return None

def parse_date(date_str):
    """
    Parse a date string into a datetime object.

    Args:
        date_str (str): Input date string to parse

    Returns:
        datetime: Parsed datetime object or None if parsing fails
    """
    obj = parse_flexible_date(date_str)
    if pd.isna(obj):
        return None
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def parse_date_list(date_list):
    """
    Parse a list of date strings into datetime objects.

    Args:
        date_list (list): List of date strings

    Returns:
        list: List of parsed datetime objects
    """
    return [parse_flexible_date(date) for date in date_list]

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point

    Returns:
        Distance in meters between the points
    """
    R = 6371000  # Earth's radius in meters

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance

def calculate_area(coordinates: List[Dict[str, float]]) -> float:
    """
    Calculate the area of a polygon defined by coordinates using the Shoelace formula

    Args:
        coordinates: List of dictionaries containing 'lat' and 'lon' keys

    Returns:
        Area in square meters
    """
    if len(coordinates) < 3:
        # Not enough points to form a polygon
        print(f"Not enough points to form a polygon {coordinates}")
        return 0.0

    # Convert geographic coordinates to meters using the first point as reference
    ref_lat = coordinates[0]['lat']
    ref_lon = coordinates[0]['lon']

    # Convert to x,y coordinates (in meters)
    points = []
    for coord in coordinates:
        # Calculate distances
        dy = haversine_distance(ref_lat, ref_lon, coord['lat'], ref_lon)
        dx = haversine_distance(ref_lat, ref_lon, ref_lat, coord['lon'])

        # Adjust signs based on direction
        if coord['lat'] < ref_lat:
            dy = -dy
        if coord['lon'] < ref_lon:
            dx = -dx

        points.append((dx, dy))

    # Calculate area using Shoelace formula
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0

    return area

def calculate_polygon_centroid(coordinates: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate the centroid of a polygon

    Args:
        coordinates: List of dictionaries containing 'lat' and 'lon' keys

    Returns:
        Dictionary with 'lat' and 'lon' of the centroid
    """
    if not coordinates:
        return None

    lat_sum = sum(coord['lat'] for coord in coordinates)
    lon_sum = sum(coord['lon'] for coord in coordinates)
    n = len(coordinates)

    return {
        'lat': lat_sum / n,
        'lon': lon_sum / n
    }