import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import re
from datetime import datetime
from shapely.geometry import Polygon
import pyproj
import hdbscan
from .api import OverpassAPI
from .utils import parse_date

class PowerDataExtractor:
    def __init__(self, api_url: Optional[str] = None):
        self.config_dir = Path(__file__).parent.parent / "config"
        self.api = OverpassAPI(api_url=api_url)

        # Load configuration
        self.config = self._load_config()
        
        # Initialize unit conversion patterns
        self.unit_patterns = {
            'power': re.compile(r'^(-?\d+\.?\d*)\s*([kKMGT]?[Ww]|MW|GW)$'),
            'length': re.compile(r'^(-?\d+\.?\d*)\s*([mk]m)$'),
            'area': re.compile(r'^(-?\d+\.?\d*)\s*([mk]m²)$'),
        }
        
        # Initialize unit conversion factors (to kW, m, m²)
        self.power_conversion = {
            'W': 0.001,
            'kW': 1,
            'MW': 1000,
            'GW': 1000000,
        }
        
        self.length_conversion = {
            'm': 1,
            'km': 1000,
        }

    def _load_config(self) -> Dict:
        """Load the configuration from current.yaml."""
        try:
            with open(self.config_dir / "keys" / "current.yaml", 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {}

    def convert_power_value(self, value: str) -> Optional[float]:
        """Convert power value to standard unit (kW)."""
        if not isinstance(value, str):
            return None
            
        match = self.unit_patterns['power'].match(value.strip())
        if match:
            number, unit = match.groups()
            unit = unit.upper()
            if unit == 'W':
                factor = self.power_conversion['W']
            elif unit == 'KW':
                factor = self.power_conversion['kW']
            elif unit == 'MW':
                factor = self.power_conversion['MW']
            elif unit == 'GW':
                factor = self.power_conversion['GW']
            else:
                return None
                
            try:
                return float(number) * factor
            except ValueError:
                return None
        return None

    def convert_length_value(self, value: str) -> Optional[float]:
        """Convert length value to standard unit (m)."""
        if not isinstance(value, str):
            if isinstance(value, (int, float)):
                return float(value)
            return None
            
        # First try to match value with unit
        match = self.unit_patterns['length'].match(value.strip())
        if match:
            number, unit = match.groups()
            try:
                return float(number) * self.length_conversion[unit]
            except (ValueError, KeyError):
                return None
        
        # If no unit pattern matches, try to convert direct numeric value
        try:
            # Remove any whitespace and try to convert to float
            return float(value.strip())
        except (ValueError, AttributeError):
            return None

    def process_element(self, element: Dict, ways_data: Dict, country_code: str) -> Dict:
        """Process a single element and extract relevant information."""
        processed = {
            'id': element.get('id'),
            'type': element.get('type'),
            'lat': element.get('lat'),
            'lon': element.get('lon'),
            'timestamp': datetime.now().isoformat(),
            'country_code': country_code
        }

        # Process tags according to configuration
        tags = element.get('tags', {})
        
        # Process each category from config
        for category in ['power_output', 'physical_dimensions', 'temporal', 'metadata']:
            if category in self.config:
                for key, config in self.config[category].items():
                    if key in tags:
                        if category in ['power_output', 'physical_dimensions']:
                            processed[f"{key}_raw"] = tags[key]
                            if category == 'power_output':
                                processed[f"{key}_value"] = self.convert_power_value(tags[key])
                                processed[f"{key}_unit"] = 'kW'
                            else:
                                processed[f"{key}_value"] = self.convert_length_value(tags[key])
                                processed[f"{key}_unit"] = 'm'
                        elif category == 'temporal':
                            processed[key] = parse_date(tags[key])
                        else:
                            processed[key] = tags[key]

        # Calculate area for ways when generator:source is solar and generator:method is photovoltaic
        if element['type'] == 'way' and str(element['id']) in ways_data:
            way_data = ways_data[str(element['id'])]
            if 'nodes' in way_data and 'generator:source' in tags and 'generator:method' in tags:
                if tags["generator:source"] == "solar" and tags["generator:method"] == "photovoltaic":
                    nodes = [node for node in way_data['nodes'].values()]
                    area = self.calculate_way_area(nodes)
                    if area is not None:
                        processed['area_m2'] = area
                        processed['generator:output:electricity_value'] = 150 * area / 1000 # kW = 150 W/m2 * area_m2/1000 TODO: hard-coded value
                        processed['generator:output:electricity_unit'] = 'kW'
                        processed['generator:output:electricity_raw'] = "Calculated: 150 W/m2 * area_m2 * 1 kW/1000 W"

            # Add center coordinates for ways
            if 'center' in way_data:
                processed['lat'] = way_data['center']['lat']
                processed['lon'] = way_data['center']['lon']

        return processed

    def calculate_way_area(self, nodes: List[Dict[str, float]]) -> Optional[float]:
        """Calculate area of a way in square meters using geodesic calculations."""
        if len(nodes) < 3:
            return None

        # Create a polygon from the coordinates
        coords = [(node['lon'], node['lat']) for node in nodes]
        
        # Ensure the polygon is closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        # Create a polygon
        poly = Polygon(coords)
        
        # Create a geodesic transformer
        geod = pyproj.Geod(ellps='WGS84')
        
        try:
            # Calculate the area
            area = abs(geod.geometry_area_perimeter(poly)[0])
            return area
        except Exception as e:
            print(f"Error calculating area: {e}")
            return None

    def extract_data(self, countries: Optional[List[str]]=None, force_refresh: bool=False) -> pd.DataFrame:
        """Extract and process all data."""
        if countries is not None:
            power_data = {}
            ways_data = {}
            for country in countries:
                country_code = self.api.get_country_code(country)
                power, ways = self.api.get_data(country, force_refresh=force_refresh)
                power_data.update({country_code: power})
                ways_data.update(ways)
        else:
            power_data, ways_data = self.api.get_all_data()

        processed_data = []
        for country_code, country_data in power_data.items():
            for element in country_data.get('elements', []):
                processed = self.process_element(element, ways_data, country_code)
                processed_data.append(processed)

        # Call the aggregate way clusters method
        cluster_summary = self.aggregate_way_clusters(ways_data)

        # Create DataFrame from the processed_data
        df = pd.DataFrame(processed_data)

        # If needed, merge the cluster_summary DataFrame with the main DataFrame (df)
        # Example merge; customize as per your requirements
        df = df.merge(cluster_summary, left_on='id', right_on='cluster', how='left')

        # Sort columns
        # df = df[self.target_columns]  # or any relevant processing if needed
        return df
    
    def aggregate_way_clusters(self, ways_data: Dict) -> pd.DataFrame:
        """
        Aggregate ways into clusters based on node coordinates.

        Parameters
        ----------
        ways_data : Dict
            The ways data containing nodes' coordinates.

        Returns
        -------
        pd.DataFrame
            DataFrame with clusters, centroid coordinates, and aggregated power.
        """
        coordinates = []
        completions = []
        for way_id, way_info in ways_data.items():
            for node_id, node_info in way_info['nodes'].items():
                coordinates.append((node_info['lon'], node_info['lat'], way_info.get('power', 0)))
                completions.append(way_id)

        # Converting coordinates for clustering
        coordinates_df = pd.DataFrame(coordinates, columns=['lon', 'lat', 'power'])

        # Apply HDBSCAN clustering
        clustering = hdbscan.HDBSCAN(min_cluster_size=5).fit(coordinates_df[['lon', 'lat']])
        coordinates_df['cluster'] = clustering.labels_

        # Calculate centroid and aggregate power and area for each cluster
        cluster_summary = (
            coordinates_df.groupby('cluster')
            .agg({'power': 'sum', 'lon': 'mean', 'lat': 'mean'})
            .reset_index()
        )
        
        cluster_summary.rename(columns={'lon': 'centroid_lon', 'lat': 'centroid_lat'}, inplace=True)
        cluster_summary.to_csv('cluster_summary.csv', index=False)

        return cluster_summary

def main():
    pass

if __name__ == "__main__":
    main()