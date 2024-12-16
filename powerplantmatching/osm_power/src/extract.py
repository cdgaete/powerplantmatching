import pandas as pd
import numpy as np
import pycountry
import logging
import inspect
import yaml
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from sklearn.cluster import DBSCAN, KMeans
from hdbscan import HDBSCAN
from pathlib import Path
from shapely.geometry import Point, Polygon
from math import radians, sin, cos, sqrt, atan2
from .api import OverpassAPI


logger = logging.getLogger(__name__)

@dataclass
class PowerSource:
    type: str
    tags: List[str]
    capacity_keys: List[str]
    clustering: Optional[Dict] = None
    estimation: Optional[Dict] = None

@dataclass
class PlantPolygon:
    id: str
    type: str
    obj: Polygon

@dataclass
class Plant:
    id: str
    type: str
    source: str
    lat: float
    lon: float
    capacity_mw: Optional[float] = None
    capacity_source: Optional[str] = None
    country: Optional[str] = None
    name: Optional[str] = None
    generator_count: Optional[int] = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

class PowerPlantExtractor:
    def __init__(self, config_dir: Optional[Path] = None, custom_config: Optional[Dict[str, Any]] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self.api = OverpassAPI()
        self.load_configurations(custom_config=custom_config)
        self.get_cache()
        self.gen_out = {}

    def get_cache(self):
        self.cache_ways = self.api._load_cache(self.api.ways_cache)
        self.cache_relations = self.api._load_cache(self.api.relations_cache)
        self.cache_nodes = self.api._load_cache(self.api.nodes_cache)

    def query_cached_element(self, element_type: str, element_id: str) -> Dict:
        try:
            if element_type == 'node':
                cached = self.cache_nodes
                if cached and element_id in cached['data']:
                    return cached['data'][element_id]
                return None
            elif element_type == 'way':
                cached = self.cache_ways
                if cached and element_id in cached['data']:
                    return cached['data'][element_id]
                return None
            elif element_type == 'relation':
                cached = self.cache_relations
                if cached and element_id in cached['data']:
                    return cached['data'][element_id]
                return None
            else:
                raise ValueError(f"Invalid element type: {element_type}")
        except Exception as e:
            raise ValueError(f"Error querying element {element_id}: {str(e)}")

    def load_configurations(self, custom_config: Optional[Dict[str, Any]] = None):
        if custom_config:
            self.sources = {
                name: PowerSource(**config)
                for name, config in custom_config["sources"].items()
            }
        else:
            with open(self.config_dir / "sources.yaml") as f:
                self.sources = {
                    name: PowerSource(**config)
                    for name, config in yaml.safe_load(f).items()
                }

    def extract_plants(self, countries: List[str], force_refresh: bool = False) -> pd.DataFrame:
        self.gen_out = {}
        all_plants = []

        for country in countries:
            plants_data, generators_data = self.api.get_country_data(
                country,
                force_refresh=force_refresh
            )

            country_obj = pycountry.countries.lookup(country)
  
            primary_plants, plant_polygons = self._process_plants(plants_data, country=country_obj.name)

            secondary_plants = self._process_generators(generators_data, plant_polygons, country=country_obj.name)

            country_plants = primary_plants + secondary_plants
            all_plants.extend(country_plants)

        return pd.DataFrame([plant.to_dict() for plant in all_plants])

    def _process_plants(self, plants_data: Dict, country: Optional[str] = None) -> Tuple[List[Plant], List[PlantPolygon]]:
        self.current = "Plants"
        processed_plants = []
        plant_polygons = []

        # orgaize elements by type starting with 'relation' then 'way' then 'node'
        plants_data['elements'] = sorted(plants_data['elements'], key=lambda x: ['relation', 'way', 'node'].index(x['type']))

        ways_in_relations = set()
        ways_rel_mapping = {}
        for element in plants_data['elements']:
            plant = self._process_plant_element(element, country=country)
            if plant:
                if element['type'] == 'relation':
                    for rel_element in element['members']:
                        if rel_element['type'] == 'way':
                            ways_in_relations.add(str(rel_element['ref']))
                            ways_rel_mapping[str(rel_element['ref'])] = str(element['id'])
                    processed_plants.append(plant)
                elif element['type'] == 'way':
                    if str(element['id']) in ways_in_relations:
                        # way is in relation. We keep relation and is skipped
                        logger.debug(f"Way {element['id']} is in relation: {ways_rel_mapping[str(element['id'])]}. Skipping...")
                        continue
                    polygon = self._create_polygon(element)
                    if polygon:
                        plant_polygons.append(polygon)
                    else:
                        logger.debug(f"Failed to create polygon for element {element['id']} of type {element['type']} with nodes {element['nodes']}")
                    processed_plants.append(plant)
                elif element['type'] == 'node':
                    processed_plants.append(plant)
            else:
                logger.debug(f"Failed to process element {element['id']} of type {element['type']}")

        return processed_plants, plant_polygons
    
    def _process_plant_element(self, element: Dict, country: Optional[str] = None) -> Optional[Plant]:

        plant_data = self._extract_plant_data(element, country=country)
        if not plant_data:
            logger.debug(f"Failed to extract data for element {element['id']} of type {element['type']}")
            return None

        if element['type'] == 'way':
            if 'capacity_mw' not in plant_data:
                way_data = self.query_cached_element('way', str(element['id']))
                if way_data and 'nodes' in way_data:
                    total_capacity = 0
                    for node_id in way_data['nodes']:
                        node_data = self.query_cached_element('node', str(node_id))
                        if node_data:
                            node_capacity, _ = self._get_plant_capacity(node_data)
                            if node_capacity:
                                total_capacity += node_capacity
                    
                    if total_capacity > 0:
                        plant_data['capacity_mw'] = total_capacity
                        plant_data['capacity_source'] = 'Aggregated'
                if 'capacity_mw' not in plant_data:
                    nodes = []
                    for node_id in way_data['nodes']:
                        node_data = self.query_cached_element('node', str(node_id))
                        if node_data:
                            nodes.append({'lon': node_data['lon'], 'lat': node_data['lat']})
                    

                    if nodes:
                        area = calculate_area(nodes)
                        source_type = plant_data['source']
                        if source_type in self.sources:
                            source_config = self.sources[source_type]
                            if source_config.estimation and source_config.estimation['method'] == 'area_based':
                                efficiency = source_config.estimation['efficiency']
                                plant_data['capacity_mw'] = (area * efficiency) / 1e6
                                plant_data['capacity_source'] = 'Estimated'

        elif element['type'] == 'relation':
            if 'capacity_mw' not in plant_data:
                relation_data = self.query_cached_element('relation', str(element['id']))
                if relation_data and 'members' in relation_data:
                    total_capacity = 0
                    for member in relation_data['members']:
                        if member['type'] == 'way':
                            # way in relation are ussually not a power plant, so skip
                            continue
                        elif member['type'] == 'node':
                            node_data = self.query_cached_element('node', str(member['ref']))
                            if node_data:
                                node_capacity, _ = self._get_plant_capacity(node_data)
                                if node_capacity:
                                    total_capacity += node_capacity
                    if total_capacity > 0:
                        plant_data['capacity_mw'] = total_capacity
                        plant_data['capacity_source'] = 'Aggregated'
        return Plant(**plant_data) if self._validate_plant_data(plant_data) else None
    
    def _create_polygon(self, element: Dict) -> Optional[PlantPolygon]:
        way_data = self.query_cached_element('way', str(element['id']))
        if way_data and 'nodes' in way_data:
            coords = []
            for node_id in way_data['nodes']:
                node_data = self.query_cached_element('node', str(node_id))
                if node_data:
                    coords.append((node_data['lon'], node_data['lat']))
            if len(coords) >= 3:
                polygon = Polygon(coords)
                plantpolygon = PlantPolygon(id=str(element['id']), type=element['type'], obj=polygon)
                return plantpolygon
            else:
                logger.debug(f"Failed to create polygon for element {element['id']} of type {element['type']} with nodes {element['nodes']}")
        return None
    
    def _extract_plant_data(self, element: Dict, country: Optional[str] = None) -> Optional[Dict]:
        plant_data = {
            'id': element['id'],
            'type': f"{element["tags"]["power"]}:{element['type']}",
            'source': element.get('tags', {}).get('plant:source', None) or element.get('tags', {}).get('generator:source', None),
            'name': element.get('tags', {}).get('name'),
            'country': country,
        }

        coords = self._get_element_coordinates(element)
        if coords:
            plant_data.update(coords)
        else:
            return None

        capacity, capacity_source = self._get_plant_capacity(element)
        if capacity:
            plant_data['capacity_mw'] = capacity
            plant_data['capacity_source'] = capacity_source

        return plant_data if self._validate_plant_data(plant_data) else None

    @staticmethod
    def _validate_plant_data(plant_data: Dict) -> bool:
        required_fields = ['id', 'type', 'source', 'lat', 'lon']
        return all(field in plant_data for field in required_fields)
    
    def _get_plant_capacity(self, element: Dict) -> Tuple[Optional[float], str]:
        tags = element.get('tags', {})

        capacity_keys = [
            'plant:output:electricity',
            'generator:output:electricity',
        ]

        for key in capacity_keys:
            if key in tags:
                normalized_capacity = self._normalize_capacity(tags[key])
                if normalized_capacity:
                    return normalized_capacity, "Direct"
                else:
                    pass

        source_type = tags.get('plant:source') or tags.get('generator:source')
        if source_type in self.sources:
            source_config = self.sources[source_type]
            if source_config.estimation:
                if source_config.estimation['method'] == 'area_based':
                    if element['type'] == 'way':
                        way_data = self.query_cached_element('way', str(element['id']))
                        if way_data and 'nodes' in way_data:
                            coords = []
                            for node_id in way_data['nodes']:
                                node_data = self.query_cached_element('node', str(node_id))
                                if node_data:
                                    coords.append({'lat': node_data['lat'], 'lon': node_data['lon']})
                            if coords:
                                area = calculate_area(coords)
                                efficiency = source_config.estimation['efficiency']
                                return (area * efficiency) / 1e6, "Estimated"
                elif source_config.estimation['method'] == 'default_value':
                    return source_config.estimation['default_capacity'] / 1000, "Estimated"

        return -10000000, "Unknown"

    def _normalize_capacity(self, capacity_str: str) -> Optional[float]:
        if not capacity_str:
            return None

        capacity_str = str(capacity_str).strip().lower()
        capacity_str = capacity_str.replace(',', '.')

        match = re.match(r'^(\d+(?:\.\d+)?)\s*([a-zA-Z]+p?)?$', capacity_str)
        if match:
            value, unit = match.groups()
            value = float(value)

            if unit in ['w', 'watts', 'wp']:
                return value / 1e6
            elif unit in ['kw', 'kilowatts', 'kwp']:
                return value / 1000
            elif unit in ['mw', 'megawatts', 'mwp']:
                return value
            elif unit in ['gw', 'gigawatts', 'gwp']:
                return value * 1000
            else:
                return value
        else:
            try:
                return float(capacity_str)
            except ValueError:
                logger.debug(f"Failed to parse capacity string: {capacity_str}")
                return None
            
    def _get_element_coordinates(self, element: Dict) -> Optional[Dict]:
        if element['type'] == 'node':
            node_data = self.query_cached_element('node', str(element['id']))
            if node_data:
                return {'lat': node_data['lat'], 'lon': node_data['lon']}
        elif element['type'] == 'way':
            way_data = self.query_cached_element('way', str(element['id']))
            if way_data and 'center' in way_data:
                return way_data['center']
        elif element['type'] == 'relation':
            relation_data = self.query_cached_element('relation', str(element['id']))
            if relation_data and 'members' in relation_data:
                coords = []
                for member in relation_data['members']:
                    if member['type'] == 'node':
                        node_data = self.query_cached_element('node', str(member['ref']))
                        if node_data:
                            coords.append({'lat': node_data['lat'], 'lon': node_data['lon']})
                    elif member['type'] == 'way':
                        way_data = self.query_cached_element('way', str(member['ref']))
                        if way_data and 'center' in way_data:
                            coords.append(way_data['center'])
                if coords:
                    return calculate_polygon_centroid(coords)
                else:
                    logger.debug(f"Failed to get coordinates for element {element}")
        else:
            logger.debug(f"Unsupported element type: {element['type']}")
        return None

    def _process_generators(self, generators_data: Dict, plant_polygons: List[Polygon], country: Optional[str] = None) -> List[Plant]:
        self.current = "Generators"
        generators_by_type = self._group_generators_by_type(generators_data['elements'])
        processed_plants = []

        for source_type, generators in generators_by_type.items():
            filtered_generators = self._filter_generators(generators, plant_polygons)
            if source_type not in self.sources:
                for generator in filtered_generators:
                    plant = self._process_plant_element(generator, country=country)
                    if plant:
                        processed_plants.append(plant)
                    else:
                        logger.debug(f"Failed to extract data for generator {generator['id']}")
            else:
                source_config = self.sources[source_type]

                if source_config.clustering:
                    clusters = self._cluster_generators(
                        filtered_generators,
                        source_config.clustering
                    )

                    for cluster in clusters:
                        plant = self._process_generator_cluster(
                            cluster,
                            source_type,
                            country=country,
                        )
                        if plant:
                            processed_plants.append(plant)

        return processed_plants

    def _process_generator_cluster(self, cluster: List[Dict], source_type: str, country: Optional[str] = None) -> Optional[Plant]:
        if not cluster:
            return None

        lats = []
        lons = []
        total_capacity = 0

        for generator in cluster:
            coords = self._get_element_coordinates(generator)
            if coords:
                lats.append(coords['lat'])
                lons.append(coords['lon'])

            capacity, _ = self._get_plant_capacity(generator)
            if capacity:
                total_capacity += capacity

        if not lats or not lons:
            logger.debug(f"Failed to get coordinates for generator cluster {cluster[0]['id']}")
            return None

        return Plant(
            id=f"cluster_{cluster[0]['id']}",
            type='generator_cluster',
            source=source_type,
            country=country,
            lat=sum(lats) / len(lats),
            lon=sum(lons) / len(lons),
            capacity_mw=total_capacity,
            capacity_source='Aggregated',
            generator_count=len(cluster)
        )

    def _group_generators_by_type(self, generators: List[Dict]) -> Dict[str, List[Dict]]:
        grouped = {}
        for generator in generators:
            source = generator.get('tags', {}).get('generator:source')
            if source:
                grouped.setdefault(source, []).append(generator)
        return grouped

    def _filter_generators(self, generators: List[Dict], plant_polygons: List[PlantPolygon]) -> List[Dict]:
        filtered_generators = []
        for generator in generators:
            coords = self._get_element_coordinates(generator)
            if coords:
                point = Point(coords['lon'], coords['lat'])
                outside_flag = True
                for plantpolygon in plant_polygons:
                    if plantpolygon.obj.contains(point):
                        logger.debug(f"Generator {generator['id']} is in the plant polygon {plantpolygon.type}/{plantpolygon.id}:{plantpolygon.obj}. Skipping...")
                        outside_flag = False
                        self.gen_out[generator['id']] = [plantpolygon.id, generator]
                        break
                if outside_flag:
                    filtered_generators.append(generator)
            else:
                logger.debug(f"Failed to get coordinates for generator {generator['id']}")
        return filtered_generators

    @staticmethod
    def _cluster_fn(fn, coords, config):
        if 'to_radians' in config:
            if config['to_radians']:
                coords = np.radians(coords)

        signature = inspect.signature(fn)
        possible_parameters = list(signature.parameters.keys())
        clustering = fn(**{param:config[param] for param in config if param in possible_parameters}).fit(coords)
        return clustering

    def _cluster_generators(self, generators: List[Dict], clustering_config: Dict) -> List[List[Dict]]:
        arrays = []
        for gen in generators:
            array = self._get_element_coordinates(gen)
            if array:
                arrays.append([array['lat'], array['lon']])
        coords = np.array(arrays)

        if clustering_config['method'] == 'kmeans':
            clustering = self._cluster_fn(KMeans, coords, clustering_config)
        elif clustering_config['method'] == 'dbscan':
            clustering = self._cluster_fn(DBSCAN, coords, clustering_config)
        elif clustering_config['method'] == 'hdbscan':
            clustering = self._cluster_fn(HDBSCAN, coords, clustering_config)
        else:
            raise ValueError(f"Unknown clustering method: {clustering_config['method']}")

        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label >= 0:
                clusters.setdefault(label, []).append(generators[i])

        return list(clusters.values())
    
# Functions

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

def main():
    extractor = PowerPlantExtractor()
    pass

if __name__ == "__main__":
    main()