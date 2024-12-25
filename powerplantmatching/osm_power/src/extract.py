from datetime import date
import pandas as pd
import numpy as np
import pycountry
import logging
import inspect
import yaml
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from sklearn.cluster import DBSCAN, KMeans
from hdbscan import HDBSCAN
from pathlib import Path
from shapely.geometry import Point, Polygon
from math import radians, sin, cos, sqrt, atan2
from .api import OverpassAPI
from .flow_analysis import FlowAnalyzer


logger = logging.getLogger(__name__)

@dataclass
class PowerSource:
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
    case: Optional[str] = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

class PowerPlantExtractor:
    def __init__(self, config_dir: Optional[Path] = None, custom_config: Optional[Dict[str, Any]] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self.load_configurations(custom_config=custom_config)
        self.api = OverpassAPI(custom_config=custom_config)
        self.gen_out = {}
        self.clusters = {}
        self.flow_analyzer = FlowAnalyzer()

    def get_cache(self):
        self.cache_ways = self.api._load_cache(self.api.ways_cache, date_check=self.api.date_check)
        self.cache_relations = self.api._load_cache(self.api.relations_cache, date_check=self.api.date_check)
        self.cache_nodes = self.api._load_cache(self.api.nodes_cache, date_check=self.api.date_check)

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
        self.clusters = {}
        all_plants = []

        for country in countries:
            plants_data, generators_data = self.api.get_country_data(
                country,
                force_refresh=force_refresh
            )

            self.get_cache()

            country_obj = pycountry.countries.lookup(country)
  
            primary_plants, plant_polygons = self._process_plants(plants_data, country=country_obj.name)

            secondary_plants = self._process_generators(generators_data, plant_polygons, country=country_obj.name)

            country_plants = primary_plants + secondary_plants
            all_plants.extend(country_plants)

        df = pd.DataFrame([plant.to_dict() for plant in all_plants])
        
        self.last_extracted_df = df

        return df

    def _process_plants(self, plants_data: Dict, country: Optional[str] = None) -> Tuple[List[Plant], List[PlantPolygon]]:
        self.current = "Plants"
        processed_plants = []
        plant_polygons = []

        # organize elements by type starting with 'relation' then 'way' then 'node'
        plants_data['elements'] = sorted(
            plants_data['elements'], 
            key=lambda x: ['relation', 'way', 'node'].index(x['type'])
        )

        ways_in_relations = set()
        ways_rel_mapping = {}
        for element in plants_data['elements']:
            # Pass the current plant_polygons list
            plant = self._process_plant_element(
                element, 
                plant_polygons=plant_polygons,  # Pass the current plant_polygons
                country=country, 
                case="plants"
            )
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
    
    def _process_plant_element(self, element: Dict, plant_polygons: Optional[List[PlantPolygon]] = None, country: Optional[str] = None, case: Optional[str] = None) -> Optional[Plant]:
        flow_path = ["Start Data Collection"]
        
        # Track element type
        flow_path.append(f"Element Type: {element['type']}")
        
        # Track power type
        power_type = element.get('tags', {}).get('power')
        flow_path.append(f"Power Type: {power_type}")
        
        if power_type == 'plant':
            flow_path.append("Process as Plant")
            
            # Track element type for plant
            flow_path.append(f"Plant Element Type: {element['type']}")
            
            # Track capacity presence
            has_capacity = False
            capacity, source = self._get_plant_capacity(element)
            if capacity is not None:
                has_capacity = True
                flow_path.append("Has Direct Capacity")
            else:
                flow_path.append("No Direct Capacity")
                
                # Track capacity estimation path
                if element['type'] == 'way':
                    source_type = element.get('tags', {}).get('plant:source')
                    if source_type == 'solar':
                        flow_path.append("Solar Area Estimation")
                    elif source_type == 'wind':
                        flow_path.append("Wind Default Value")
                    else:
                        flow_path.append("No Capacity Available")
                elif element['type'] == 'relation':
                    flow_path.append("Check Member Elements")
                    
        elif power_type == 'generator':
            flow_path.append("Process as Generator")
            
            # Track if inside plant polygon
            coords = self._get_element_coordinates(element)
            if coords and plant_polygons:  # Check if plant_polygons is provided
                point = Point(coords['lon'], coords['lat'])
                inside_plant = False
                for plantpolygon in plant_polygons:
                    if plantpolygon.obj.contains(point):
                        inside_plant = True
                        flow_path.append("Inside Plant Polygon")
                        break
                if not inside_plant:
                    flow_path.append("Outside Plant Polygon")
                    
                    # Track source type presence
                    source_type = element.get('tags', {}).get('generator:source')
                    if source_type:
                        flow_path.append("Has Source Type")
                        if source_type in self.sources:
                            flow_path.append("Source in Config")
                        else:
                            flow_path.append("Source not in Config")
                    else:
                        flow_path.append("No Source Type")
        
        # Record the flow path
        self.flow_analyzer.add_flow(" -> ".join(flow_path))
        
        # Original processing logic continues here...
        plant_data = self._extract_plant_data(element, country=country, case=case)
        if not plant_data:
            return None

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
    
    def _extract_plant_data(self, element: Dict, country: Optional[str] = None, case: Optional[str] = None) -> Optional[Dict]:
        plant_data = {
            'id': element['id'],
            'type': f"{element["tags"]["power"]}:{element['type']}",
            'source': element.get('tags', {}).get('plant:source', None) or element.get('tags', {}).get('generator:source', None),
            'name': element.get('tags', {}).get('name'),
            'country': country,
            'case': case,
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
                    return normalized_capacity, "direct"

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
                                return (area * efficiency) / 1e6, "estimated"
                elif source_config.estimation['method'] == 'default_value':
                    return source_config.estimation['default_capacity'] / 1000, "estimated"

        return None, "unknown"

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
                return value/1000 # If no unit is specified, assume to kilowatts
        else:
            try:
                return float(capacity_str)/1000
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

    def _process_generators(self, generators_data: Dict, plant_polygons: List[PlantPolygon], country: Optional[str] = None) -> List[Plant]:
        self.current = "Generators"
        generators_by_type = self._group_generators_by_type(generators_data['elements'])
        processed_plants = []

        for source_type, generators in generators_by_type.items():
            filtered_generators = self._filter_generators(generators, plant_polygons)
            if source_type not in self.sources:
                for generator in filtered_generators:
                    plant = self._process_plant_element(
                        generator, 
                        plant_polygons=plant_polygons,
                        country=country, 
                        case="excluded_source"
                    )
                    if plant:
                        processed_plants.append(plant)
                    else:
                        logger.debug(f"Failed to extract data for generator {generator['id']}")
            else:
                source_config = self.sources[source_type]

                if source_config.clustering:
                    labels = self._cluster_generators(
                        filtered_generators,
                        source_config.clustering
                    )

                    cluster_bin = {}
                    units = []
                    for i, label in enumerate(labels):
                        if label >= 0:
                            cluster_bin.setdefault(label, []).append(generators[i])
                        else:
                            units.append(generators[i])

                    clusters = list(cluster_bin.values())

                    if units:
                        for generator in units:
                            plant = self._process_plant_element(
                                generator, 
                                plant_polygons=plant_polygons,
                                country=country, 
                                case="noise_point"
                            )
                            if plant:
                                processed_plants.append(plant)
                            else:
                                logger.debug(f"Failed to extract data for generator {generator['id']}")

                    # Prepare and visualize cluster data
                    viz_data = self.prepare_cluster_visualization_data(filtered_generators, labels)
                    if country not in self.clusters:
                        self.clusters[country] = {}
                    self.clusters[country][source_type] = viz_data

                    if not clusters:
                        logger.warning(f"No clusters found for source type: {source_type}")
                        continue

                    for cluster in clusters:
                        plant = self._process_generator_cluster(
                            cluster,
                            clusters.index(cluster),
                            source_type,
                            country=country,
                            case="cluster_point"
                        )

                        processed_plants.append(plant)

        return processed_plants

    def _process_generator_cluster(self, cluster: List[Dict], index: int, source_type: str, country: Optional[str] = None, case: Optional[str] = None) -> Optional[Plant]:
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
            id=f"cluster_{cluster[0]['type']}_{cluster[0]['id']}",
            type='generator:cluster',
            source=source_type,
            country=country,
            lat=sum(lats) / len(lats),
            lon=sum(lons) / len(lons),
            capacity_mw=total_capacity,
            capacity_source='aggregated',
            generator_count=len(cluster),
            name=f"cluster_{str(index).zfill(3)}_{country}_{source_type}",
            case=case
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
        model = fn(**{param:config[param] for param in config if param in possible_parameters})
        clustering = model.fit(coords)
        return clustering

    def _cluster_generators(self, generators: List[Dict], clustering_config: Dict) -> Tuple[List[List[Dict]], np.ndarray]:
        arrays = []
        for gen in generators:
            array = self._get_element_coordinates(gen)
            if array:
                arrays.append([array['lat'], array['lon']])
        
        if not arrays:
            logger.warning(f"No generators found for clustering with config: {clustering_config}")
            return np.array([])
        
        coords = np.array(arrays)

        if clustering_config['method'] == 'kmeans':
            clustering = self._cluster_fn(KMeans, coords, clustering_config)
        elif clustering_config['method'] == 'dbscan':
            clustering = self._cluster_fn(DBSCAN, coords, clustering_config)
        elif clustering_config['method'] == 'hdbscan':
            clustering = self._cluster_fn(HDBSCAN, coords, clustering_config)
        else:
            raise ValueError(f"Unknown clustering method: {clustering_config['method']}")
        
        return clustering.labels_

    def prepare_cluster_visualization_data(self, generators: List[Dict], labels: np.ndarray) -> Dict[str, List]:
        data = {
            'lat': [],
            'lon': [],
            'cluster': [],
            'capacity': [],
            'id': []
        }
        
        for i, generator in enumerate(generators):
            coords = self._get_element_coordinates(generator)
            if coords:
                data['lat'].append(coords['lat'])
                data['lon'].append(coords['lon'])
                data['cluster'].append(int(labels[i]))
                capacity, _ = self._get_plant_capacity(generator)
                data['capacity'].append(capacity if capacity is not None else 0)
                data['id'].append(generator['id'])
        
        return data
    
    def plot_clusters(self, country: str = 'all', source_type: str = 'all', show: bool = False):
        combined_data = self._combine_cluster_data(country, source_type)
        
        if combined_data.empty:
            raise ValueError("No data available for the specified country and/or source type")

        fig = go.Figure()

        for (country, source, cluster), group in combined_data.groupby(['country', 'source', 'cluster']):
            fig.add_trace(
                go.Scattermapbox(
                    lat=group['lat'],
                    lon=group['lon'],
                    mode='markers',
                    marker=dict(size=10),
                    text=group['text'],
                    name=f'{country}:{source}:{cluster}'
                )
            )

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(
                    lat=combined_data['lat'].mean(),
                    lon=combined_data['lon'].mean()
                ),
                zoom=3
            ),
            showlegend=True,
            height=800,
            width=1200,
            title_text="Generator Clusters"
        )
        
        if show:
            fig.show(config={'scrollZoom': True})
        
        return fig

    def _combine_cluster_data(self, country: str = 'all', source_type: str = 'all') -> pd.DataFrame:
        combined_data = []

        for c, country_data in self.clusters.items():
            if country != 'all' and c != country:
                continue
            for s, source_data in country_data.items():
                if source_type != 'all' and s != source_type:
                    continue
                df = pd.DataFrame(source_data)
                df['country'] = c
                df['source'] = s
                df['text'] = df.apply(lambda row: f"Country: {c}<br>Source: {s}<br>ID: {row['id']}<br>Capacity: {row['capacity']:.2f} MW", axis=1)
                combined_data.append(df)

        return pd.concat(combined_data, ignore_index=True) if combined_data else pd.DataFrame()
    
    def summary(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate a summary of power capacity by country and source.
        
        Args:
            df (pd.DataFrame, optional): The dataframe to summarize. If None, uses the last extracted data.
        
        Returns:
            pd.DataFrame: A summary dataframe with power capacity by country and source, including totals.
        """
        if df is None:
            if not hasattr(self, 'last_extracted_df'):
                raise ValueError("No data available. Please run extract_plants() first or provide a dataframe.")
            df = self.last_extracted_df

        df['capacity_mw'] = pd.to_numeric(df['capacity_mw'], errors='coerce')
        summary = df.groupby(['country', 'source'])['capacity_mw'].sum().unstack(fill_value=0)
        summary['Total'] = summary.sum(axis=1)
        source_totals = summary.sum().to_frame('Total').T
        summary = pd.concat([summary, source_totals])
        summary = summary.round(2)
        summary = summary.sort_values('Total', ascending=False)
        return summary
    
    def get_flow_summary(self) -> pd.DataFrame:
        """Get a summary of the data processing flows."""
        return self.flow_analyzer.get_summary()

    def plot_flow_sankey(self, title: str = "Data Processing Flow") -> go.Figure:
        """Create a Sankey diagram of the data processing flows."""
        return self.flow_analyzer.plot_sankey(title)

    def plot_flow_sunburst(self, title: str = "Data Processing Flow Distribution") -> go.Figure:
        """Create a sunburst diagram of the data processing flows."""
        return self.flow_analyzer.plot_sunburst(title)


def visualize_clusters(data: Dict[str, List], title: str = "Generator Clusters", show: bool = False):
    fig = go.Figure()

    # Create a scatter plot for each cluster
    for cluster in set(data['cluster']):
        cluster_data = [i for i in range(len(data['cluster'])) if data['cluster'][i] == cluster]
        
        fig.add_trace(go.Scattermapbox(
            lat=[data['lat'][i] for i in cluster_data],
            lon=[data['lon'][i] for i in cluster_data],
            mode='markers',
            marker=dict(size=10),
            text=[f"ID: {data['id'][i]}<br>Capacity: {data['capacity'][i]:.2f} MW" for i in cluster_data],
            name=f'Cluster {cluster}'
        ))

    # Update the layout
    fig.update_layout(
        title=title,
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=np.mean(data['lat']), lon=np.mean(data['lon'])),
            zoom=10
        ),
        showlegend=True
    )

    if show:
        fig.show(config={'scrollZoom': True})

    return fig
    
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