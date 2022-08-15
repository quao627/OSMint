# OSMint

This is a Python package for extracting signalized intersections from OpenStreetMap. We use Overpass API to collect raw data about traffic signals, road segments, and turn restrictions from OpenStreetMap. Then, the package generates a signalized intersection dataset through a pipeline with imputation mechanisms for various missing values (lane count, speed limit, turns, gradient, etc) and algorithms for detecting turns and combining one-ways that should have been a divided two-way. An example output for representing a road intersection is shown below.
![img](./example.png)
