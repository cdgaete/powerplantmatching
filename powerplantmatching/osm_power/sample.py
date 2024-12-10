from src.extract import PowerDataExtractor
import json

# list of countries
countries = ["New Zealand", "Uruguay", "Luxembourg", "Chile", "Congo", "Nigeria", "Myanmar", "Ethiopia", "Tanzania"]
# list of code alpha2
codes = ["NZ", "UY", "LU", "CL", "CG", "NG", "MM", "ET", "TZ"]


# Initialize extractor
extractor = PowerDataExtractor()

# Extract and process data both cases works
df = extractor.extract_data(countries)
# or 
df = extractor.extract_data(codes)

# Save to CSV
df.head().to_csv('osm_power/power_facilities.csv', index=False)

unique_values = {}
for column in df.columns:
    if column not in ['lat', 'lon', 'id', 'timestamp', 'area_m2', 'generator:output:electricity_value', 
                      'generator:output:electricity_raw', 'plant:output:electricity_raw', 
                      'plant:output:electricity_value','name', 'source', 'operator', 'start_date']:
        unique_values[column] = df[column].unique().tolist()

with open('osm_power/unique_values.json', 'w') as f:
    json.dump(unique_values, f, indent=4)
