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

    # Extract OSM data for a specific country
    osm_data = pm.data.OSM(country=['Portugal', 'Spain'])

    # View the extracted data
    print(osm_data.head())

    # Get a summary of the extracted data
    summary = pm.utils.lookup(osm_data)
    print(summary)

    # Combine with other data sources
    combined_data = pm.data.OPSD().powerplant.combine([osm_data])

    # Match the combined data
    matched_data = pm.matching.match(combined_data)

This example demonstrates how to extract OSM data for a specific country, view and summarize the data, and then integrate it with other data sources in powerplantmatching.

Conclusion
----------

The OSM integration provides a powerful addition to the powerplantmatching tool, offering flexible and efficient access to OpenStreetMap power plant data. By leveraging this integration, users can enhance their power plant analyses with up-to-date and community-driven geographical information. The customizable configuration options and efficient processing mechanisms make it a valuable resource for a wide range of power system studies and analyses.
