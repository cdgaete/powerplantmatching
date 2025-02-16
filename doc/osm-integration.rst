OpenStreetMap Integration
=========================

Overview
--------

This module integrates OpenStreetMap (OSM) data into the powerplantmatching tool, providing a robust and flexible way to extract and process power plant information from OSM. It enhances the tool's capabilities by adding a comprehensive, community-driven data source for power plant locations and characteristics.

Main Features
-------------

The OSM integration offers several key features to enhance powerplantmatching:

1. **OSM Data Extraction**: Retrieve power plant data from OpenStreetMap for specific countries or regions.
2. **Flexible Configuration**: Customize data extraction and processing through various configuration options.
3. **Power Plant Focus**: Option to focus solely on power plant data, excluding individual generators.
4. **Capacity Estimation**: Estimate power plant capacities based on area or default values when not provided.
5. **Clustering**: Group nearby generators into single power plant entities (optional).
6. **Caching Mechanism**: Efficient data storage and retrieval using JSON cache files.
7. **Integration with powerplantmatching**: Seamlessly use OSM data alongside other data sources in the tool.

These features allow users to tailor the OSM data extraction to their specific needs, whether for broad regional analyses or detailed local studies.

Configuration Options
---------------------

The OSM integration can be customized through the ``config.yaml`` file. Here's an example configuration with explanations:

.. code-block:: yaml

    OSM:
      reliability_score: 3
      net_capacity: false
      api_url: https://overpass-api.de/api/interpreter
      fn: osm_data.csv
      force_refresh: false
      plants_only: true
      enable_estimation: false
      enable_clustering: false
      sources:
        solar:
          clustering:
            method: dbscan
            to_radians: false
            eps: 0.003
            min_samples: 2
          estimation:
            method: area_based
            efficiency: 150
        wind:
          clustering:
            method: dbscan
            to_radians: false
            eps: 0.009
            min_samples: 2
          estimation:
            method: default_value
            default_capacity: 2000

Configuration Parameters Explained
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``reliability_score`` indicates the reliability of OSM data compared to other sources, on a scale of 1 to 5. ``net_capacity`` determines whether capacities are interpreted as net (true) or gross (false).

The ``api_url`` specifies the Overpass API endpoint for querying OSM data, while ``fn`` sets the filename for saving extracted OSM data locally.

``force_refresh``, when set to true, forces a new data download even if a cache exists. ``plants_only``, when true, extracts only power plant data, excluding individual generators.

``enable_estimation`` allows capacity estimation for plants without capacity information, and ``enable_clustering`` enables grouping of nearby generators into single power plants.

The ``sources`` section provides configuration options for different power source types, such as solar and wind. Each source type can have its own clustering and estimation settings.

Source Configuration
^^^^^^^^^^^^^^^^^^^^

The ``sources`` section in the configuration allows you to specify settings for different types of power plants. This is particularly useful for customizing the extraction and processing of specific energy sources. Let's break down the structure:

.. code-block:: yaml

    sources:
      solar:
        clustering:
          method: dbscan
          to_radians: false
          eps: 0.003
          min_samples: 2
        estimation:
          method: area_based
          efficiency: 150
      wind:
        clustering:
          method: dbscan
          to_radians: false
          eps: 0.009
          min_samples: 2
        estimation:
          method: default_value
          default_capacity: 2000

For each source (e.g., solar, wind), you can define:

1. **Clustering settings**: Used to group nearby generators into single power plant entities.
   - ``method``: The clustering algorithm (currently supports DBSCAN).
   - ``to_radians``: Whether to convert coordinates to radians before clustering.
   - ``eps``: The maximum distance between two samples for them to be considered as in the same neighborhood.
   - ``min_samples``: The number of samples in a neighborhood for a point to be considered as a core point.

2. **Estimation settings**: Used to estimate capacity when it's not provided in the OSM data.
   - ``method``: The estimation method (e.g., 'area_based' for solar, 'default_value' for wind).
   - ``efficiency`` or ``default_capacity``: Parameters specific to the estimation method.

This configuration allows for fine-tuned control over how different types of power plants are processed, accommodating the unique characteristics of each energy source.

Default Configuration: plants_only Mode
---------------------------------------

The recommended default configuration focuses on power plants only:

.. code-block:: yaml

    OSM:
      plants_only: true
      enable_estimation: false
      enable_clustering: false

This configuration extracts only power plant data, ignoring individual generators. It also disables capacity estimation and clustering for faster processing, providing raw OSM data without modifications or assumptions.

The benefits of this configuration include reduced data volume, faster processing, and lower memory usage. It provides an accurate representation of OSM power plant data. However, users should be aware that this configuration results in coarser granularity, and missing details about individual generators.

Implementation Details
----------------------

Caching Mechanism
^^^^^^^^^^^^^^^^^

The OSM integration uses JSON files for efficient caching. Separate JSON cache files are maintained for plants, generators, ways, nodes, and relations. These cache files preserve complex data structures and types, significantly improving performance for repeated queries. The caching mechanism is managed by the ``OverpassAPI`` class, which handles saving and loading data as needed.

Geometry Processing Hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Power plant geometries are processed in a specific order to ensure accurate representation and avoid duplication. The process begins with relations, which often represent the most complex and complete power plant structures. For each relation, a polygon is created from all its member ways, and a centroid is calculated to represent the plant's location. All member ways and nodes of the relation are then marked as "processed" to avoid duplication.

After relations, standalone ways (not part of any processed relation) are handled. For each way, if it's not already part of a processed relation, a polygon is created from its nodes, and a centroid is calculated. The way and all its nodes are then marked as processed.

Finally, standalone nodes (not part of any processed relation or way) are processed. These are typically used for smaller or simpler power plant representations.

Centroid Calculation and Entity Discarding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Centroids are used to represent power plants as single points, ensuring a consistent representation across different OSM geometry types. The centroid calculation method varies based on the geometry's complexity, ranging from simple averages for basic polygons to area-weighted centroids for more complex shapes.

To prevent duplication, ways and nodes that are part of processed relations are discarded as standalone entities. This ensures that each plant is represented only once in the final dataset.

Polygons play a crucial role in this process. They're essential for accurate centroid calculation, especially for large or irregularly shaped plants. In non-``plants_only`` mode, polygons are also used to determine if generators fall within a plant's boundaries. While centroids are used as the primary location reference, the original polygon data is often retained for potential future use or detailed analysis.

Usage Example
-------------

Here's a basic example of how to use the OSM integration in powerplantmatching:

.. code-block:: python

    import powerplantmatching as pm

    # Setting OSM configurations
    config = pm.get_config()

    config["main_query"] = ""
    config["target_countries"] = ["Uruguay", "Paraguay"]
    config["OSM"]["plants_only"] = True
    config["OSM"]["fn"] = "osm_plants.csv"
    config["OSM"]["reliability_score"] = 6
    config["matching_sources"] = {"OSM": None, "GEM": None}   

    # Extract OSM data for a specific country
    osm_data = pm.data.OSM(update=False, config=config)

    # View the extracted data
    print(osm_data.head())

    # Plot Power Plants
    fig = osm_data.powerplant.plotly_map()
    fig.update_layout(height=800, width=1200)
    fig.show(config={'scrollZoom': True})

    # Match the combined data
    data = pm.powerplants(update=True, config_update=config)

Line-by-line explanation:

1. Import the powerplantmatching library.
2. Get the default configuration.
3. Set the main query to an empty string (no additional filtering).
4. Specify the target countries for data extraction.
5. Set `plants_only` to True to focus on power plants and exclude individual generators.
6. Specify the filename for saving the extracted OSM data.
7. Set the reliability score for the OSM data source.
8. Specify the matching sources to be used (OSM and GEM in this case).
9. Extract OSM data using the specified configuration.
10. Print the first few rows of the extracted data to inspect it.
11-13. Create an interactive map of the power plants using plotly.
14. Perform the matching process with the updated configuration, including OSM data.

This example demonstrates how to extract OSM data for specific countries, view and summarize the data, and then integrate it with other data sources in powerplantmatching.

Additional Features and Functionalities
---------------------------------------

1. Customizing clustering settings:

.. code-block:: python

    config["OSM"]["enable_clustering"] = True
    config["OSM"]["sources"]["solar"]["clustering"] = {
        "method": "dbscan",
        "eps": 0.005,
        "min_samples": 3
    }
    osm_data = pm.data.OSM(update=True, config=config)

This snippet enables clustering for solar power plants and customizes the DBSCAN parameters.

2. Enabling capacity estimation:

.. code-block:: python

    config["OSM"]["enable_estimation"] = True
    config["OSM"]["sources"]["wind"]["estimation"] = {
        "method": "default_value",
        "default_capacity": 2500  # in kW
    }
    osm_data = pm.data.OSM(update=True, config=config)

This enables capacity estimation for wind power plants using a default value.

3. Visualizing clusters (if clustering is enabled):

.. code-block:: python

    extractor = pm.osm.PowerPlantExtractor(custom_config=config)
    extractor.extract_plants(["Uruguay", "Paraguay"])
    cluster_plot = extractor.plot_clusters(country="Uruguay", source_type="solar", show=True)

This visualizes the clusters of solar power plants in Uruguay.

4. Combining OSM data with other sources:

.. code-block:: python

    config["matching_sources"] = {"OSM": None, "OPSD": None, "GEM": None}
    combined_data = pm.powerplants(update=True, config_update=config)
    print(combined_data.head())

This combines OSM data with OPSD and GEM data sources in the matching process.

5. Exporting the data:

.. code-block:: python

    osm_data.to_csv("osm_power_plants.csv", index=False)

This exports the extracted OSM data to a CSV file.

Benefits of OSM Integration
---------------------------

1. **Comprehensive Data Source**: OSM provides a vast, community-driven dataset of power plants worldwide.
2. **Regular Updates**: OSM data is frequently updated by contributors, ensuring relatively current information.
3. **Flexible Extraction**: Users can extract data for specific countries or regions as needed.
4. **Customizable Processing**: The configuration options allow users to tailor the data extraction and processing to their specific needs.
5. **Integration with Existing Sources**: OSM data can be seamlessly combined with other power plant databases in powerplantmatching.
6. **Improved Coverage**: OSM can provide information on power plants that might be missing from other sources, especially for smaller or newer installations.
7. **Open Data**: OSM is an open data source, which aligns well with open science principles and reproducibility.

Conclusion
----------

The OSM integration provides a powerful addition to the powerplantmatching tool, offering flexible and efficient access to OpenStreetMap power plant data. By leveraging this integration, users can enhance their power plant analyses with up-to-date and community-driven geographical information. The customizable configuration options and efficient processing mechanisms make it a valuable resource for a wide range of power system studies and analyses.
