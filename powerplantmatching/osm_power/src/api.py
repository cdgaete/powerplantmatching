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
        self.countries_cache = os.path.join(self.raw_dir, "countries_power.json")
        self.ways_cache = os.path.join(self.raw_dir, "ways_data.json")

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

    def get_power_data(self, country: str, force_refresh: bool = False) -> Dict:
        """Get power generation data for a country."""
        country_code = self.get_country_code(country)
        
        # Load cache
        if not force_refresh:
            cached = self._load_cache(self.countries_cache)
            if cached and country_code in cached['data']:
                return cached['data'][country_code]

        # Main power facilities query
        query = f"""
        [out:json][timeout:300];
        area["ISO3166-1"="{country_code}"][admin_level=2]->.boundaryarea;
        (
            node["power"="plant"](area.boundaryarea);
            way["power"="plant"](area.boundaryarea);
            relation["power"="plant"](area.boundaryarea);
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
            cached = self._load_cache(self.countries_cache) or {'data': {}}
            cached['data'][country_code] = data
            self._save_cache(self.countries_cache, cached['data'])
            
            return data
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch data: {str(e)}")

    def get_ways_data(self, way_ids: List[int]) -> Dict:
        """Get centroids and nodes for a list of ways in a single query."""
        # Load cache
        cached = self._load_cache(self.ways_cache) or {'data': {}}
        
        # Filter out already cached ways that were cached today
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
    
    def get_data(self, country: str, force_refresh: bool = False) -> Dict:
        """Get power data for a country."""
        power_data = self.get_power_data(country, force_refresh)
        ways_data = self.get_ways_data([element['id'] for element in power_data['elements']])
        return power_data, ways_data
    
    def get_all_data(self, force_refresh: bool = False) -> Dict:
        """Get power data for all countries already cached."""
        if not force_refresh:
            cached = self._load_cache(self.countries_cache)
            if cached:
                ways_all = {'data': {}}
                for _, power_data in cached.get('data', {}).items():
                    ways = self.get_ways_data([element['id'] for element in power_data['elements']])
                    ways_all['data'].update(ways)
                return cached, ways_all
            else:
                raise ValueError("No cached data found")
        else:
            # get coutry codes
            cached = self._load_cache(self.countries_cache) or {'data': {}}
            countries = list(cached.get('data', {}).keys())
            ways_all = {'data': {}}
            for country in countries:
                power_data, ways_data = self.get_data(country, force_refresh)
                ways_all['data'].update(ways_data)
                cached['data'][country] = power_data
            self._save_cache(self.countries_cache, cached['data'])
            return cached, ways_all

def main():
    # Test the API functionality
    api = OverpassAPI()

    try:
        # Test with Luxembourg
        print("Fetching data for Luxembourg...")
        data = api.get_power_data("Luxembourg")
        print(f"Found {len(data['elements'])} power facilities")

        # Collect all way IDs
        way_ids = [
            element['id'] 
            for element in data['elements'] 
            if element['type'] == 'way'
        ]

        if way_ids:
            print(f"\nFetching data for {len(way_ids)} ways...")
            ways_data = api.get_ways_data(way_ids)
            print(f"Retrieved data for {len(ways_data)} ways")
            
            # Print example data for the first way
            first_way_id = str(way_ids[0])
            if first_way_id in ways_data:
                way_data = ways_data[first_way_id]
                print(f"\nExample data for way {first_way_id}:")
                print(f"Center: {way_data['center']}")
                print(f"Number of nodes: {len(way_data['nodes'])}")
                if way_data['nodes']:
                    first_node = next(iter(way_data['nodes'].values()))
                    print(f"Sample node: {first_node}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()