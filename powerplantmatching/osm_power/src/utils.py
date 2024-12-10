import pandas as pd
from datetime import datetime
import re
import numpy as np

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