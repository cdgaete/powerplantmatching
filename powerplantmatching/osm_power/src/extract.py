from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import inspect
import yaml
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from hdbscan import HDBSCAN
from .api import OverpassAPI
from .utils import parse_date, calculate_area
import re

@dataclass
class PowerSource:
    """Configuration for a power source type"""
    type: str
    tags: List[str]
    capacity_keys: List[str]
    clustering: Optional[Dict] = None
    estimation: Optional[Dict] = None

class PowerPlantExtractor:
    """Main class for extracting power plant data"""

    def __init__(self, config_dir: Optional[Path] = None, custom_config: Optional[Dict[str, Any]] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self.api = OverpassAPI()
        self.load_configurations(custom_config=custom_config)
        self.generators_omitted = []
        self.relations_omitted = []
        self.plants_omitted = []

    def load_configurations(self, custom_config: Optional[Dict[str, Any]] = None):
        """Load all configuration files"""
        if custom_config:
            self.sources = {
                name: PowerSource(**config)
                for name, config in custom_config["sources"].items()
            }
        else:
            # Load source configurations
            with open(self.config_dir / "sources.yaml") as f:
                self.sources = {
                    name: PowerSource(**config)
                    for name, config in yaml.safe_load(f).items()
                }

    def _get_plant_capacity(self, element: Dict, ways_data: Optional[Dict] = None) -> Optional[float]:
        """
        Extract and normalize capacity value from a plant element.
        
        Args:
            element: Dictionary containing plant data
            ways_data: Optional dictionary containing way node data
            
        Returns:
            Normalized capacity in MW or None if not available
        """
        tags = element.get('tags', {})
        
        # Check for direct capacity values
        capacity_keys = [
            'plant:output:electricity',
            'generator:output:electricity',
            'capacity',
            'power_output'
        ]
        
        for key in capacity_keys:
            if key in tags:
                return self._normalize_capacity(tags[key])
        
        # If no direct capacity, try to estimate based on source type
        source_type = tags.get('plant:source') or tags.get('generator:source')
        if source_type in self.sources:
            source_config = self.sources[source_type]
            if source_config.estimation:
                if source_config.estimation['method'] == 'area_based':
                    # Calculate area and apply efficiency factor
                    if element['type'] == 'way' and ways_data:
                        way_data = ways_data.get(str(element['id']))
                        if way_data and 'nodes' in way_data:
                            # Get coordinates from ways_data
                            coords = [
                                {'lat': node_data['lat'], 'lon': node_data['lon']}
                                for node_id, node_data in way_data['nodes'].items()
                            ]
                            if coords:
                                area = calculate_area(coords)
                                efficiency = source_config.estimation['efficiency']  # W/m2
                                return (area * efficiency) / 1e6  # Convert W to MW
                elif source_config.estimation['method'] == 'default_value':
                    return source_config.estimation['default_capacity'] / 1000  # Convert kW to MW
        
        return None

    def _extract_plant_data(self, element: Dict, ways_data: Dict) -> Optional[Dict]:
        """Extract data from a single plant element"""
        # Get basic properties
        plant_data = {
            'id': element['id'],
            'type': element['type'],
            'source': element.get('tags', {}).get('plant:source'),
            'name': element.get('tags', {}).get('name'),
        }

        # Get coordinates
        coords = self._get_element_coordinates(element, ways_data)
        if coords:
            plant_data.update(coords)

        # Get capacity
        capacity = self._get_plant_capacity(element, ways_data)
        if capacity:
            plant_data['capacity_mw'] = capacity

        return plant_data if self._validate_plant_data(plant_data) else None

    def _process_generator_cluster(self, cluster: List[Dict], source_type: str,
                                source_config: PowerSource, ways_data: Dict) -> Optional[Dict]:
        """Process a cluster of generators into a single plant"""
        if not cluster:
            return None

        # Calculate cluster centroid
        lats = []
        lons = []
        total_capacity = 0

        for generator in cluster:
            coords = self._get_element_coordinates(generator, ways_data)
            if coords:
                lats.append(coords['lat'])
                lons.append(coords['lon'])
            
            # Sum up capacities
            capacity = self._get_plant_capacity(generator, ways_data)
            if capacity:
                total_capacity += capacity

        if not lats or not lons:
            return None

        return {
            'id': f"cluster_{cluster[0]['id']}",
            'type': 'generator_cluster',
            'source': source_type,
            'lat': sum(lats) / len(lats),
            'lon': sum(lons) / len(lons),
            'capacity_mw': total_capacity,
            'generator_count': len(cluster)
        }

    def _process_generators(self, generators_data: Dict, ways_data: Dict) -> List[Dict]:
        """Process and cluster generator data"""
        generators_by_type = self._group_generators_by_type(generators_data['elements'])
        processed_plants = []

        for source_type, generators in generators_by_type.items():
            if source_type not in self.sources:
                continue

            source_config = self.sources[source_type]

            if source_config.clustering:
                # Cluster generators into plants
                clusters = self._cluster_generators(
                    generators,
                    ways_data,
                    source_config.clustering
                )

                # Process each cluster as a plant
                for cluster in clusters:
                    plant = self._process_generator_cluster(
                        cluster,
                        source_type,
                        source_config,
                        ways_data  # Pass ways_data to the method
                    )
                    if plant:
                        processed_plants.append(plant)

        return processed_plants

    def _normalize_capacity(self, capacity_str: str) -> Optional[float]:
        """
        Normalize capacity string to MW.
        
        Args:
            capacity_str: String containing capacity value and optional unit
            
        Returns:
            Capacity in MW or None if parsing fails
        """
        if not capacity_str:
            return None
            
        # Convert to string if not already
        capacity_str = str(capacity_str).strip().lower()
        
        # Try to extract number and unit using regex
        match = re.match(r'^(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?$', capacity_str)
        if not match:
            try:
                # Try direct conversion if just a number
                return float(capacity_str) / 1000  # Assume kW if no unit
            except ValueError:
                return None
                
        value, unit = match.groups()
        value = float(value)
        
        # Convert to MW based on unit
        if unit in ['w', 'watts']:
            return value / 1e6
        elif unit in ['kw', 'kilowatts']:
            return value / 1000
        elif unit in ['mw', 'megawatts']:
            return value
        elif unit in ['gw', 'gigawatts']:
            return value * 1000
        
        # Default to kW if unit not recognized
        return value / 1000

    def extract_plants(self, countries: List[str], force_refresh: bool = False) -> pd.DataFrame:
        """Main method to extract power plant data"""
        all_plants = []

        for country in countries:
            # Fetch raw data
            plants_data, generators_data, ways_data = self.api.get_country_data(
                country,
                force_refresh=force_refresh
            )

            # Process primary sources (plants)
            primary_plants = self._process_plants(plants_data, ways_data)

            # Process secondary sources (generators)
            secondary_plants = self._process_generators(generators_data, ways_data)

            # Merge and deduplicate
            country_plants = self._merge_plants(primary_plants, secondary_plants)
            all_plants.extend(country_plants)

        return pd.DataFrame(all_plants)

    def _process_plants(self, plants_data: Dict, ways_data: Dict) -> List[Dict]:
        """Process direct power plant data"""
        processed_plants = []

        for element in plants_data['elements']:
            if element['type'] not in ['way', 'node', 'relation']:
                continue

            plant = self._extract_plant_data(element, ways_data)
            if plant:
                processed_plants.append(plant)

        return processed_plants
    
    @staticmethod
    def _cluster_fn(fn, coords, config):
        """Helper function to perform clustering. Coordinates lat/lon are in degrees and can be converted to radians."""
        if 'to_radians' in config:
            if config['to_radians']:
                coords = np.radians(coords)

        signature = inspect.signature(fn)
        possible_parameters = list(signature.parameters.keys())
        clustering = fn(**{param:config[param] for param in config if param in possible_parameters}).fit(coords)
        return clustering


    def _cluster_generators(self, generators: List[Dict], ways_data: Dict,
                            clustering_config: Dict) -> List[List[Dict]]:
        """Cluster generators based on configuration. Coordinates lat/lon are in degrees converted to radians. 
        Distance related parameters should be in radians."""
        # Extract coordinates for clustering
        arrays = []
        for gen in generators:
            array = self._get_element_coordinates(gen, ways_data)
            if array:
                arrays.append([array['lat'], array['lon']])
        coords = np.array(arrays)

        # Perform clustering
        if clustering_config['method'] == 'kmeans':
            
            clustering = self._cluster_fn(KMeans, coords, clustering_config)

        elif clustering_config['method'] == 'dbscan':
            clustering = self._cluster_fn(DBSCAN, coords, clustering_config)

        elif clustering_config['method'] == 'hdbscan':
            clustering = self._cluster_fn(HDBSCAN, coords, clustering_config)

        else:
            raise ValueError(f"Unknown clustering method: {clustering_config['method']}")

        # Group generators by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label >= 0:  # Ignore noise points
                clusters.setdefault(label, []).append(generators[i])

        return list(clusters.values())

    def _merge_plants(self, primary_plants: List[Dict],
                      secondary_plants: List[Dict]) -> List[Dict]:
        """Merge and deduplicate plants"""
        all_plants = primary_plants + secondary_plants

        # Implement deduplication logic based on spatial proximity
        # and other rules from configuration

        return all_plants

    @staticmethod
    def _validate_plant_data(plant_data: Dict) -> bool:
        """Validate extracted plant data"""
        required_fields = ['id', 'type', 'source']
        return all(field in plant_data for field in required_fields)

    @staticmethod
    def _get_element_coordinates(element: Dict, ways_data: Dict) -> Optional[Dict]:
        """Get coordinates for an element"""
        if element['type'] == 'node':
            return {'lat': element['lat'], 'lon': element['lon']}
        elif element['type'] == 'way':
            way_data = ways_data.get(str(element['id']))
            if way_data and 'center' in way_data:
                return way_data['center']
        return None

    def _group_generators_by_type(self, generators: List[Dict]) -> Dict[str, List[Dict]]:
        """Group generators by their source type"""
        grouped = {}
        for generator in generators:
            source = generator.get('tags', {}).get('generator:source')
            if source:
                grouped.setdefault(source, []).append(generator)
        return grouped


def main():
    extractor = PowerPlantExtractor()
    # Add test code here
    pass

if __name__ == "__main__":
    main()