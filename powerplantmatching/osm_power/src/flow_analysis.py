from collections import defaultdict
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FlowAnalyzer:
    """Analyzes and visualizes the data processing flow paths."""
    
    def __init__(self):
        self.flow_counts = defaultdict(int)
        self.total_elements = 0
        
    def add_flow(self, path: str):
        """Record a flow path."""
        self.flow_counts[path] += 1
        self.total_elements += 1
    
    def get_percentages(self) -> Dict[str, float]:
        """Calculate percentages for each flow path."""
        if self.total_elements == 0:
            return {}
        
        return {
            path: (count / self.total_elements) * 100
            for path, count in self.flow_counts.items()
        }
    
    def get_summary(self) -> pd.DataFrame:
        """Generate a summary DataFrame of flow paths and their percentages."""
        percentages = self.get_percentages()
        
        df = pd.DataFrame({
            'Path': list(percentages.keys()),
            'Count': [self.flow_counts[path] for path in percentages.keys()],
            'Percentage': list(percentages.values())
        })
        
        df = df.sort_values('Count', ascending=False)
        df['Percentage'] = df['Percentage'].round(2)
        
        return df
    
    def plot_sankey(self, title: str = "Data Processing Flow") -> go.Figure:
        """Create a Sankey diagram of the flow paths."""
        # Parse flow paths into nodes and links
        all_nodes = set()
        links = defaultdict(int)
        
        for path, count in self.flow_counts.items():
            steps = path.split(' -> ')
            for i in range(len(steps) - 1):
                source, target = steps[i], steps[i+1]
                all_nodes.add(source)
                all_nodes.add(target)
                links[(source, target)] += count
        
        # Create node labels and mapping
        node_labels = sorted(list(all_nodes))
        node_map = {node: idx for idx, node in enumerate(node_labels)}
        
        # Prepare Sankey data
        sankey_data = {
            'node': {'label': node_labels},
            'link': {
                'source': [],
                'target': [],
                'value': [],
                'label': []
            }
        }
        
        for (source, target), value in links.items():
            sankey_data['link']['source'].append(node_map[source])
            sankey_data['link']['target'].append(node_map[target])
            sankey_data['link']['value'].append(value)
            sankey_data['link']['label'].append(f'{value} ({(value/self.total_elements*100):.1f}%)')
        
        # Create figure
        fig = go.Figure(data=[go.Sankey(
            node=sankey_data['node'],
            link=sankey_data['link']
        )])
        
        fig.update_layout(
            title=title,
            font_size=10,
            height=800
        )
        
        return fig
    
    def plot_sunburst(self, title: str = "Data Processing Flow Distribution") -> go.Figure:
        """Create a sunburst diagram of the flow paths."""
        # Create nested dictionary structure
        def nested_dict():
            return defaultdict(nested_dict)
        
        data_tree = nested_dict()
        
        # Build tree structure
        for path, count in self.flow_counts.items():
            steps = path.split(' -> ')
            current = data_tree
            for step in steps:
                current = current[step]
            current['_value'] = count  # Use _value to avoid conflicts with subdictories
        
        # Prepare data for sunburst
        ids = []
        labels = []
        parents = []
        values = []
        
        def process_tree(tree, parent_id=""):
            for key, subtree in tree.items():
                if key == '_value':
                    continue
                    
                # Create ID for current node
                current_id = f"{parent_id}_{key}" if parent_id else key
                
                # Add node information
                ids.append(current_id)
                labels.append(key)
                parents.append(parent_id)
                
                # Calculate value (sum of all nested _values)
                def sum_values(t):
                    total = t.get('_value', 0)
                    for k, v in t.items():
                        if k != '_value' and isinstance(v, dict):
                            total += sum_values(v)
                    return total
                
                values.append(sum_values(subtree))
                
                # Process children
                process_tree(subtree, current_id)
        
        process_tree(data_tree)
        
        # Create figure
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total"
        ))
        
        fig.update_layout(
            title=title,
            width=800,
            height=800
        )
        
        return fig