#!/usr/bin/env python3
"""
Module to collect power plant data from OpenStreetMap using overpy.
"""

import logging
import os
import pandas as pd
import overpy
from tqdm import tqdm

from .core import _data_in, get_config
from .utils import set_column_name

logger = logging.getLogger(__name__)

def get_powerplants_query(country_area):
    """
    Returns the Overpass query for power plants in a given country area.
    
    Parameters
    ----------
    country_area : str
        The OSM area ID for the country (e.g., "3600062594" for Germany)
    """
    return f"""
        area({country_area})->.searchArea;
        (
          way["power"="plant"](area.searchArea);
          relation["power"="plant"](area.searchArea);
          node["power"="plant"](area.searchArea);
          way["power"="generator"](area.searchArea);
          relation["power"="generator"](area.searchArea);
          node["power"="generator"](area.searchArea);
        );
        out body;
        >;
        out skel qt;
    """

def parse_powerplant(element):
    """
    Parse an OSM element into a power plant dictionary.
    
    Parameters
    ----------
    element : overpy.Element
        OSM element (node, way, or relation)
    
    Returns
    -------
    dict
        Dictionary containing power plant information
    """
    tags = element.tags
    
    # Basic information
    plant = {
        'Name': tags.get('name', ''),
        'Fueltype': tags.get('plant:source', tags.get('generator:source', '')),
        'Technology': tags.get('plant:method', tags.get('generator:method', '')),
        'Capacity': float(tags.get('plant:output:electricity', 
                                 tags.get('generator:output:electricity', 0))),
        'Set': 'PP',  # Default to power plant
        'projectID': f"OSM-{element.id}",
    }
    
    # Get coordinates
    if hasattr(element, 'lat') and hasattr(element, 'lon'):
        plant['lat'] = float(element.lat)
        plant['lon'] = float(element.lon)
    elif hasattr(element, 'center_lat') and hasattr(element, 'center_lon'):
        plant['lat'] = float(element.center_lat)
        plant['lon'] = float(element.center_lon)
    else:
        plant['lat'] = None
        plant['lon'] = None
    
    # Additional information
    plant['DateIn'] = tags.get('start_date', '')
    plant['DateOut'] = tags.get('end_date', '')
    plant['DateRetrofit'] = ''
    
    return plant

def collect_osm_powerplants(country_areas, update=False, config=None):
    """
    Collect power plant data from OpenStreetMap for specified countries.
    
    Parameters
    ----------
    country_areas : dict
        Dictionary mapping country names to OSM area IDs
    update : bool, default False
        Whether to force update the data
    config : dict, default None
        Configuration dictionary
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing power plant information
    """
    if config is None:
        config = get_config()
        
    output_file = _data_in('osm_powerplants.csv')
    
    # Check if we need to update
    if not update and os.path.exists(output_file):
        return pd.read_csv(output_file)
    
    api = overpy.Overpass()
    plants = []
    
    for country, area_id in tqdm(country_areas.items(), desc="Collecting OSM data"):
        try:
            # Get power plants for this country
            query = get_powerplants_query(area_id)
            result = api.query(query)
            
            # Process nodes
            for element in result.nodes:
                if 'power' in element.tags:
                    plant = parse_powerplant(element)
                    plant['Country'] = country
                    plants.append(plant)
            
            # Process ways
            for element in result.ways:
                if 'power' in element.tags:
                    plant = parse_powerplant(element)
                    plant['Country'] = country
                    plants.append(plant)
                    
            # Process relations
            for element in result.relations:
                if 'power' in element.tags:
                    plant = parse_powerplant(element)
                    plant['Country'] = country
                    plants.append(plant)
                    
        except overpy.exception.OverpassError as e:
            logger.error(f"Error collecting data for {country}: {str(e)}")
            continue
            
    # Create DataFrame
    df = pd.DataFrame(plants)
    
    # Save to file
    df.to_csv(output_file, index=False)
    
    return df.pipe(set_column_name, "OSM")

def OSM(raw=False, update=False, config=None):
    """
    Importer for OpenStreetMap power plant data.
    
    Parameters
    ----------
    raw : bool, default False
        Whether to return the raw dataset
    update : bool, default False
        Whether to update the data from OSM
    config : dict, default None
        Custom configuration
    """
    if config is None:
        config = get_config()
        
    # Define country areas (example - should be moved to config)
    country_areas = {
        "Germany": "3600051477",
        "France": "3600071427",
        # Add more countries and their OSM area IDs
    }
    
    df = collect_osm_powerplants(country_areas, update=update, config=config)
    
    if raw:
        return df
        
    # Process the data according to powerplantmatching standards
    df = (df
          .pipe(lambda x: x.assign(
              Capacity=lambda df: pd.to_numeric(df.Capacity, errors='coerce'),
              DateIn=lambda df: pd.to_numeric(df.DateIn, errors='coerce'),
              DateOut=lambda df: pd.to_numeric(df.DateOut, errors='coerce'),
          ))
          .pipe(set_column_name, "OSM")
          .pipe(config_filter, config)
    )
    
    return df