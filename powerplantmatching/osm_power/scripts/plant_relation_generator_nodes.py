import json
from collections import Counter

plants = "/home/cgaete/.cache/osm-power/raw/plants_power.json"
generators = "/home/cgaete/.cache/osm-power/raw/generators_power.json"

# Read JSON files
with open(plants, 'r') as f:
    plants_data = json.load(f)
with open(generators, 'r') as f:
    generators_data = json.load(f)

country_codes = list(plants_data["data"].keys())

for country in country_codes:
    print(country)
    nodes_data = {}
    for element in generators_data["data"][country]["elements"]:
        if element["type"] == "node":
            nodes_data[element["id"]] = element

    print(plants_data.keys())

    print(plants_data["data"].keys())

    print(plants_data["data"][country].keys())

    nodes_list = []
    tags_list = []
    for element in plants_data["data"][country]["elements"]:
        if element["type"] == "relation":
            for member in element["members"]:
                if member["type"] == "node":
                    ref = member["ref"]
                    if ref in nodes_data:
                        for tag in ["generator:source", "generator:method"]:
                            if tag in nodes_data[ref]["tags"]:
                                nodes_list.append(nodes_data[ref]["tags"][tag])
                                tags_list.append(tag)
                                break
    nodes_summary = Counter(nodes_list)
    tags_summary = Counter(tags_list)
    print(nodes_summary)
    print(tags_summary)

# Output:
# NR
# dict_keys(['timestamp', 'data'])
# dict_keys(['NR', 'UY', 'DE'])
# dict_keys(['version', 'generator', 'osm3s', 'elements'])
# Counter()
# Counter()
# UY
# dict_keys(['timestamp', 'data'])
# dict_keys(['NR', 'UY', 'DE'])
# dict_keys(['version', 'generator', 'osm3s', 'elements'])
# Counter({'wind': 668, 'hydro': 17})
# Counter({'generator:source': 685})
# DE
# dict_keys(['timestamp', 'data'])
# dict_keys(['NR', 'UY', 'DE'])
# dict_keys(['version', 'generator', 'osm3s', 'elements'])
# Counter({'wind': 12444, 'solar': 74, 'hydro': 9, 'biogas': 4, 'gas': 3, 'battery': 1, 'biomass': 1})
# Counter({'generator:source': 12536})