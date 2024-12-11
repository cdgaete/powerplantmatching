import os
import json
import requests
from typing import Dict, Optional, List
from datetime import datetime
import appdirs
import pycountry

class OverpassAPI:
    def __init__(self, api_url: Optional[str] = None):
        """
        Initialize the OverpassAPI class.

        Args:
            api_url (str, optional): The URL of the Overpass API. Defaults to https://overpass-api.de/api/interpreter.

        Returns:
            None
        """
        self.cache_dir = appdirs.user_cache_dir("osm-power")
        self.raw_dir = os.path.join(self.cache_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)

        if api_url:
            self.api_url = api_url
        else:
            # Default Overpass API endpoint
            self.api_url = "https://overpass-api.de/api/interpreter"
        
        # Cache file paths
        self.plants_cache = os.path.join(self.raw_dir, "plants_power.json")
        self.generators_cache = os.path.join(self.raw_dir, "generators_power.json")
        self.ways_cache = os.path.join(self.raw_dir, "ways_data.json")
        self.nodes_cache = os.path.join(self.raw_dir, "nodes_data.json")

    def _load_cache(self, cache_path: str) -> Optional[Dict[str,Dict]]:
        """Load cached data if it exists."""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                    # Check if cache is from today
                    if cached.get('timestamp'):
                        cache_date = datetime.fromisoformat(cached['timestamp']).date()
                        if cache_date == datetime.now().date():
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
            cached = self._load_cache(self.plants_cache)
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
            cached = self._load_cache(self.plants_cache) or {'data': {}}
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
            cached = self._load_cache(self.generators_cache)
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
            cached = self._load_cache(self.generators_cache) or {'data': {}}
            cached['data'][country_code] = data
            self._save_cache(self.generators_cache, cached['data'])
            
            return data
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch generators data: {str(e)}")

    def get_ways_data(self, way_ids: List[int]) -> Dict:
        """Get centroids and nodes for a list of ways in a single query."""
        # Load cache
        cached = self._load_cache(self.ways_cache) or {'data': {}}
        
        # Filter out already cached ways
        ways_to_fetch = [
            way_id for way_id in way_ids 
            if str(way_id) not in cached['data']
        ]
        
        if ways_to_fetch:
            # Build query for multiple ways
            ways_str = ','.join(map(str, ways_to_fetch))
            query = f"""
            [out:json];
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
                
                # Initialize way data structures
                way_nodes = {str(way_id): {'nodes': {}} for way_id in ways_to_fetch}
                
                # First pass: collect all nodes
                nodes_data = {}
                for element in new_data['elements']:
                    if element['type'] == 'node':
                        nodes_data[element['id']] = {
                            'lat': element['lat'],
                            'lon': element['lon']
                        }
                
                # Second pass: process ways and their centers
                for element in new_data['elements']:
                    if element['type'] == 'way':
                        way_id = str(element['id'])
                        # Add center data
                        if 'center' in element:
                            way_nodes[way_id]['center'] = {
                                'lat': element['center']['lat'],
                                'lon': element['center']['lon']
                            }
                        # Add nodes data
                        for node_id in element.get('nodes', []):
                            if node_id in nodes_data:
                                way_nodes[way_id]['nodes'][str(node_id)] = nodes_data[node_id]
                
                # Update cache with new data
                for way_id, way_data in way_nodes.items():
                    cached['data'][way_id] = way_data
                
                self._save_cache(self.ways_cache, cached['data'])
            
            except requests.RequestException as e:
                raise ConnectionError(f"Failed to fetch ways data: {str(e)}")
        
        # Return requested ways data
        return {
            str(way_id): cached['data'].get(str(way_id))
            for way_id in way_ids
        }

    def get_country_data(self, country: str, force_refresh: bool = False) -> Dict:
        """Get both plants and generators data for a specific country."""
        plants_data = self.get_plants_data(country, force_refresh)
        generators_data = self.get_generators_data(country, force_refresh)
        
        # Collect all way IDs from both datasets
        way_ids = []
        for dataset in [plants_data, generators_data]:
            way_ids.extend([
                element['id'] 
                for element in dataset['elements'] 
                if element['type'] == 'way'
            ])
        
        ways_data = self.get_ways_data(way_ids)
        return plants_data, generators_data, ways_data

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
        all_ways_data = {}

        # If no countries specified, return all cached data
        if countries is None or len(countries) == 0:
            # Load cached data
            plants_cached = self._load_cache(self.plants_cache)
            generators_cached = self._load_cache(self.generators_cache)
            ways_cached = self._load_cache(self.ways_cache)

            if not any([plants_cached, generators_cached, ways_cached]):
                raise ValueError("No cached data available and no countries specified")

            return {
                'plants_data': plants_cached.get('data', {}) if plants_cached else {},
                'generators_data': generators_cached.get('data', {}) if generators_cached else {},
                'ways_data': ways_cached.get('data', {}) if ways_cached else {}
            }

        # Process specified countries
        for country in countries:
            try:
                country_code = self.get_country_code(country)
                
                # Get data for current country
                plants_data, generators_data, ways_data = self.get_country_data(
                    country, 
                    force_refresh=force_refresh
                )

                # Add to collected data
                all_plants_data[country_code] = plants_data
                all_generators_data[country_code] = generators_data
                all_ways_data.update(ways_data)  # ways_data is already a dict of way_id: way_data

            except Exception as e:
                print(f"Error processing country {country}: {str(e)}")
                continue

        if not any([all_plants_data, all_generators_data, all_ways_data]):
            raise ValueError("No data could be retrieved for the specified countries")

        return {
            'plants_data': all_plants_data,
            'generators_data': all_generators_data,
            'ways_data': all_ways_data
        }

def main():
    pass

if __name__ == "__main__":
    main()