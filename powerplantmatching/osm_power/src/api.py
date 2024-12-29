import os
import json
import requests
from typing import Dict, Optional, List, Tuple, Set, Any
from datetime import datetime
import appdirs
import pycountry
import logging

logger = logging.getLogger(__name__)

class OverpassAPI:
    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OverpassAPI class.

        Args:
            custom_config (Dict[str, Any], optional): Custom configuration dictionary.

        Returns:
            None
        """
        self.config = custom_config or {}
        
        self.cache_dir = self.config.get('cache_dir', appdirs.user_cache_dir("osm-power"))
        self.raw_dir = os.path.join(self.cache_dir, "OSM")
        os.makedirs(self.raw_dir, exist_ok=True)

        self.api_url = self.config.get('api_url', "https://overpass-api.de/api/interpreter")
        self.date_check = self.config.get('date_check', False)
        self.force_refresh = self.config.get('force_refresh', False)
        
        # Cache file paths
        self.plants_cache = os.path.join(self.raw_dir, "plants_power.json")
        self.generators_cache = os.path.join(self.raw_dir, "generators_power.json")
        self.ways_cache = os.path.join(self.raw_dir, "ways_data.json")
        self.nodes_cache = os.path.join(self.raw_dir, "nodes_data.json")
        self.relations_cache = os.path.join(self.raw_dir, "relations_data.json")

    def _load_cache(self, cache_path: str, date_check: bool = False) -> Optional[Dict[str,Dict]]:
        """Load cached data if it exists."""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                    if date_check:
                        # Check cache date
                        if cached.get('timestamp'):
                            cache_date = datetime.fromisoformat(cached['timestamp']).date()
                            if cache_date.month == datetime.now().month:
                                return cached
                    return cached
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _save_cache(self, cache_path: str, data: Dict):
        """Save data to cache."""
        with open(cache_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'data': data
            }, f, indent=2)

    def get_country_code(self, country: str) -> str:
        """Get country code from country name."""
        try:
            country_obj = pycountry.countries.lookup(country)
            return country_obj.alpha_2
        except LookupError:
            raise ValueError(f"Invalid country name: {country}")

    def get_plants_data(self, country: str, force_refresh: bool = False) -> Dict:
        """Get power plant data for a country."""
        country_code = self.get_country_code(country)
        
        # Load cache
        if not force_refresh:
            cached = self._load_cache(self.plants_cache, self.date_check)
            if cached and country_code in cached['data']:
                return cached['data'][country_code]

        # Power plants query
        query = f"""
        [out:json][timeout:300];
        area["ISO3166-1"="{country_code}"][admin_level=2]->.boundaryarea;
        (
            node["power"="plant"](area.boundaryarea);
            way["power"="plant"](area.boundaryarea);
            relation["power"="plant"](area.boundaryarea);
        );
        out body;
        """

        try:
            response = requests.post(self.api_url, data={'data': query})
            response.raise_for_status()
            data = response.json()
            
            # Update cache
            cached = self._load_cache(self.plants_cache, self.date_check) or {'data': {}}
            cached['data'][country_code] = data
            self._save_cache(self.plants_cache, cached['data'])
            
            return data
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch plants data: {str(e)}")

    def get_generators_data(self, country: str, force_refresh: bool = False) -> Dict:
        """Get power generator data for a country."""
        country_code = self.get_country_code(country)
        
        # Load cache
        if not force_refresh:
            cached = self._load_cache(self.generators_cache, self.date_check)
            if cached and country_code in cached['data']:
                return cached['data'][country_code]

        # Power generators query
        query = f"""
        [out:json][timeout:300];
        area["ISO3166-1"="{country_code}"][admin_level=2]->.boundaryarea;
        (
            node["power"="generator"](area.boundaryarea);
            way["power"="generator"](area.boundaryarea);
            relation["power"="generator"](area.boundaryarea);
        );
        out body;
        """

        try:
            response = requests.post(self.api_url, data={'data': query})
            response.raise_for_status()
            data = response.json()
            
            # Update cache
            cached = self._load_cache(self.generators_cache, self.date_check) or {'data': {}}
            cached['data'][country_code] = data
            self._save_cache(self.generators_cache, cached['data'])
            
            return data
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch generators data: {str(e)}")

    def get_nodes_data(self, node_ids: List[int]) -> Dict:
        """Get data for a list of nodes."""
        nodes_cached = self._load_cache(self.nodes_cache, self.date_check) or {'data': {}}
        
        # Filter out already cached nodes
        nodes_to_fetch = [
            node_id for node_id in node_ids 
            if str(node_id) not in nodes_cached['data']
        ]
        
        if nodes_to_fetch:
            # Build query for multiple nodes
            nodes_str = ','.join(map(str, nodes_to_fetch))
            query = f"""
            [out:json][timeout:300];
            node(id:{nodes_str});
            out body;
            """

            try:
                response = requests.post(self.api_url, data={'data': query})
                response.raise_for_status()
                new_data = response.json()
                
                # Process nodes
                for element in new_data['elements']:
                    if element['type'] == 'node':
                        nodes_cached['data'][str(element['id'])] = element
                
                # Save updated cache
                self._save_cache(self.nodes_cache, nodes_cached['data'])
            
            except requests.RequestException as e:
                raise ConnectionError(f"Failed to fetch nodes data: {str(e)}")
        
        # Return requested nodes data
        return {
            str(node_id): nodes_cached['data'].get(str(node_id))
            for node_id in node_ids
        }

    def get_ways_data(self, way_ids: List[int]) -> Tuple[Dict, Set[int]]:
        """Get full data for a list of ways in a single query."""
        # Load cache
        ways_cached = self._load_cache(self.ways_cache, self.date_check) or {'data': {}}
        nodes_cached = self._load_cache(self.nodes_cache, self.date_check) or {'data': {}}
        
        # Filter out already cached ways
        ways_to_fetch = [
            way_id for way_id in way_ids 
            if str(way_id) not in ways_cached['data']
        ]
        
        if ways_to_fetch:
            # Build query for multiple ways
            ways_str = ','.join(map(str, ways_to_fetch))
            query = f"""
            [out:json][timeout:300];
            (
                way(id:{ways_str});
                >;  // Get all nodes for ways
            );
            out body;
            way(id:{ways_str});
            out center;
            """

            try:
                response = requests.post(self.api_url, data={'data': query})
                response.raise_for_status()
                new_data = response.json()
                
                node_ids = []
                # Process nodes and ways
                for element in new_data['elements']:
                    if element['type'] == 'node':
                        node_id = str(element['id'])
                        if node_id not in nodes_cached['data']:
                            node_ids.append(element['id'])
                    elif element['type'] == 'way':
                        way_id = str(element['id'])
                        ways_cached['data'][way_id] = element
                if node_ids:
                    self.get_nodes_data(node_ids)
                
                # Save updated caches
                self._save_cache(self.ways_cache, ways_cached['data'])
            
            except requests.RequestException as e:
                raise ConnectionError(f"Failed to fetch ways data: {str(e)}")
        
        # Return requested ways data and all node IDs
        return {
            str(way_id): ways_cached['data'].get(str(way_id))
            for way_id in way_ids
        }
    
    def get_relations_data(self, relation_ids: List[int]) -> Dict:
        """Get data for a list of relations."""
        relations_cached = self._load_cache(self.relations_cache, self.date_check) or {'data': {}}
        ways_cached = self._load_cache(self.ways_cache, self.date_check) or {'data': {}}
        nodes_cached = self._load_cache(self.nodes_cache, self.date_check) or {'data': {}}
        
        # Filter out already cached relations
        relations_to_fetch = [
            relation_id for relation_id in relation_ids 
            if str(relation_id) not in relations_cached['data']
        ]
        
        if relations_to_fetch:
            # Build query for multiple relations
            relations_str = ','.join(map(str, relations_to_fetch))
            query = f"""
            [out:json][timeout:300];
            relation(id:{relations_str});
            out center;
            """

            try:
                response = requests.post(self.api_url, data={'data': query})
                response.raise_for_status()
                new_data = response.json()
                
                # Process relations
                for element in new_data['elements']:
                    if element['type'] == 'relation':
                        relations_cached['data'][str(element['id'])] = element
                        nodes_ids = []
                        ways_ids = []
                        for member in element['members']:
                            if member['type'] == 'node':
                                node_id = str(member['ref'])
                                if node_id not in nodes_cached['data']:
                                    nodes_ids.append(node_id)
                            elif member['type'] == 'way':
                                way_id = str(member['ref'])
                                if way_id not in ways_cached['data']:
                                    ways_ids.append(way_id)
                        if nodes_ids:
                            self.get_nodes_data(nodes_ids)
                        if ways_ids:
                            self.get_ways_data(ways_ids)

                # Save updated cache
                self._save_cache(self.relations_cache, relations_cached['data'])
            
            except requests.RequestException as e:
                raise ConnectionError(f"Failed to fetch relations data: {str(e)}")
        
        # Return requested relations data
        return {
            str(relation_id): relations_cached['data'].get(str(relation_id))
            for relation_id in relation_ids
        }

    def get_country_data(self, country: str, force_refresh: bool = False) -> Dict:
        """Get both plants and generators data for a specific country."""
        plants_data = self.get_plants_data(country, force_refresh)
        generators_data = self.get_generators_data(country, force_refresh)
        
        # Collect all way IDs from both datasets
        way_ids = []
        relation_ids = []
        node_ids = []
        for dataset in [plants_data, generators_data]:
            way_ids.extend([
                element['id'] 
                for element in dataset['elements'] 
                if element['type'] == 'way'
            ])
            relation_ids.extend([
                element['id'] 
                for element in dataset['elements'] 
                if element['type'] == 'relation'
            ])
            node_ids.extend([
                element['id'] 
                for element in dataset['elements'] 
                if element['type'] == 'node'
            ])

        self.get_nodes_data(node_ids)
        self.get_ways_data(way_ids)
        self.get_relations_data(relation_ids)
        
        return plants_data, generators_data

    def get_countries_data(self, countries: Optional[List[str]] = None, force_refresh: bool = False) -> Dict:
        """
        Get data for multiple countries, using cache when available.
        
        Args:
            countries: List of country names. If None, returns all cached data.
            force_refresh: If True, forces download of data even if cached.
            
        Returns:
            Dict containing plants_data, generators_data, and ways_data for requested countries.
        """
        all_plants_data = {}
        all_generators_data = {}

        # If no countries specified, return all cached data
        if countries is None or len(countries) == 0:
            # Load cached data
            plants_cached = self._load_cache(self.plants_cache, self.date_check)
            generators_cached = self._load_cache(self.generators_cache, self.date_check)

            if not any([plants_cached, generators_cached]):
                raise ValueError("No cached data available and no countries specified")

            return {
                'plants_data': plants_cached.get('data', {}) if plants_cached else {},
                'generators_data': generators_cached.get('data', {}) if generators_cached else {},
            }

        # Process specified countries
        for country in countries:
            try:
                country_code = self.get_country_code(country)
                
                # Get data for current country
                plants_data, generators_data = self.get_country_data(
                    country, 
                    force_refresh=force_refresh
                )

                # Add to collected data
                all_plants_data[country_code] = plants_data
                all_generators_data[country_code] = generators_data

            except Exception as e:
                logger.warning(f"Error processing country {country}: {str(e)}")
                continue

        if not any([all_plants_data, all_generators_data]):
            raise ValueError("No data could be retrieved for the specified countries")

        return {
            'plants_data': all_plants_data,
            'generators_data': all_generators_data,
        }

def main():
    pass

if __name__ == "__main__":
    main()