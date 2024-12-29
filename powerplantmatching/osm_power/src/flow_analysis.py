from collections import defaultdict
from typing import Dict, List, Optional
import pandas as pd

class FlowAnalyzer:
    """Analyzes the data processing flow paths."""
    
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
    
    def get_sankey_data(self) -> Dict[str, List]:
        """Prepare data for Sankey diagram."""
        all_nodes = set()
        links = defaultdict(int)
        
        for path, count in self.flow_counts.items():
            steps = path.split(' -> ')
            for i in range(len(steps) - 1):
                source, target = steps[i], steps[i+1]
                all_nodes.add(source)
                all_nodes.add(target)
                links[(source, target)] += count
        
        node_labels = sorted(list(all_nodes))
        node_map = {node: idx for idx, node in enumerate(node_labels)}
        
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
        
        return sankey_data
    
    def get_sunburst_data(self) -> Dict[str, List]:
        """Prepare data for Sunburst diagram."""
        def nested_dict():
            return defaultdict(nested_dict)
        
        data_tree = nested_dict()
        
        for path, count in self.flow_counts.items():
            steps = path.split(' -> ')
            current = data_tree
            for step in steps:
                current = current[step]
            current['_value'] = count
        
        ids = []
        labels = []
        parents = []
        values = []
        
        def process_tree(tree, parent_id=""):
            for key, subtree in tree.items():
                if key == '_value':
                    continue
                    
                current_id = f"{parent_id}_{key}" if parent_id else key
                
                ids.append(current_id)
                labels.append(key)
                parents.append(parent_id)
                
                def sum_values(t):
                    total = t.get('_value', 0)
                    for k, v in t.items():
                        if k != '_value' and isinstance(v, dict):
                            total += sum_values(v)
                    return total
                
                values.append(sum_values(subtree))
                
                process_tree(subtree, current_id)
        
        process_tree(data_tree)
        
        return {
            'ids': ids,
            'labels': labels,
            'parents': parents,
            'values': values
        }

    def plot_sankey(self, title: str = "Data Processing Flow"):
        """Create a Sankey diagram of the flow paths."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly not installed. Please install plotly to use visualization features.")
        
        sankey_data = self.get_sankey_data()
        
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
    
    def plot_sunburst(self, title: str = "Data Processing Flow Distribution"):
        """Create a sunburst diagram of the flow paths."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly not installed. Please install plotly to use visualization features.")
        
        sunburst_data = self.get_sunburst_data()
        
        fig = go.Figure(go.Sunburst(
            ids=sunburst_data['ids'],
            labels=sunburst_data['labels'],
            parents=sunburst_data['parents'],
            values=sunburst_data['values'],
            branchvalues="total"
        ))
        
        fig.update_layout(
            title=title,
            width=800,
            height=800
        )
        
        return fig