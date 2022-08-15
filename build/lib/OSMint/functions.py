from geopy.distance import geodesic
import pyproj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from keplergl import KeplerGl
import requests
import osmnx as ox
from copy import deepcopy
import json
import pickle
import itertools
import networkx as nx
import geopandas as gpd
import scipy
import random
from time import sleep
from collections import defaultdict
import re
import os
import shapely
from collections import Counter
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import cascaded_union
import osmnx.projection as projection


useful_tags_path = ['name', 'lanes', 'turn:lanes:forward', 'turn:lanes:backward', 'lanes:both_ways', 'turn:lanes', 'maxspeed', 'highway']
# set the required information from osmnx
ox.utils.config(useful_tags_way=useful_tags_path)

import warnings
warnings.filterwarnings('ignore')

def get_traffic_signals(city="Salt Lake City", admin_level=8, polygon=None, boundary=None):
    overpass_url = "http://overpass-api.de/api/interpreter"
    if polygon:
        print("yes")
        coords = " ".join([x for coords in polygon.exterior.coords[:-1] for x in str(coords)[1:-1].split(", ")[::-1]])
        overpass_query = f"""
    [out:json];
    (node[highway=traffic_signals](poly:"{coords}");
     node[crossing=traffic_signals](poly:"{coords}");
     );
    out body;
    """
    else:
        overpass_query = f"""
    [out:json];
    area[name="{city}"][admin_level={admin_level}]->.a;
    (node[highway=traffic_signals](area.a);
     node[crossing=traffic_signals](area.a););
    out body;
    """

    response = requests.get(overpass_url,
                            params={'data': overpass_query})
    data = response.json()

    signals = data["elements"]
    signals = pd.DataFrame(signals)
    points = gpd.GeoSeries([Point(x,y) for x, y in zip(signals['lon'], signals['lat'])])
    signals = gpd.GeoDataFrame(signals[['id', 'lon', 'lat']], geometry=points)
    signals.crs = {'init': 'epsg:4326'}
    if boundary:
        signals = signals[signals["geometry"].apply(lambda x: x.within(boundary))]
    signals = signals.to_crs({'init': 'epsg:3395'})
    signals["x"] = signals["geometry"].apply(lambda x: x.coords[0][0])
    signals["y"] = signals["geometry"].apply(lambda x: x.coords[0][1])
    return signals.reset_index().drop("index", axis=1)

def merge(sets):
    merged = True
    while merged:
        merged = False
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = True
                    common |= x
            results.append(common)
        sets = results
    return sets

def find_clusters(signals, threshold=100):
    coords = signals[['x', 'y']].to_numpy()
    dist_matrix = scipy.spatial.distance.cdist(coords, coords)
    x_list, y_list = np.where(dist_matrix <= threshold)
    positions = np.stack((x_list, y_list), axis=1)
    first_col = positions[:, 0]
    second_col = positions[:, 1]
    neighbors = [set(second_col[np.where(first_col == i)]) for i in range(len(coords))]
    return merge(neighbors)

def get_turn_restrictions(city="Salt Lake City", admin_level=8, polygon=None, boundary=None):
    def get_geometry(data):
        if data["type"]=="way":
            geom = LineString([[x["lon"], x["lat"]] for x in data["geometry"]])
        else:
            geom = Point([data["lon"], data["lat"]])
        return geom


    overpass_url = "http://overpass-api.de/api/interpreter"
    if polygon:
        coords = " ".join([x for coords in polygon.exterior.coords[:-1] for x in str(coords)[1:-1].split(", ")[::-1]])
        overpass_query = f"""
            [out:json];
            (rel[restriction](poly:"{coords}"););
            out body geom;
            """
    else:
        overpass_query = f"""
        [out:json];
        area[name="{city}"][admin_level={admin_level}]->.a;
        (rel[restriction](area.a););
        out body geom;
        """
    response = requests.get(overpass_url,
                        params={'data': overpass_query})
    restrictions = response.json()["elements"]
    restrictions_dict = {"type": [], "id": [], "from": [], "via": [], "to": [], "from_type": [],
                         "via_type": [], "to_type": [], "from_geom": [], "via_geom": [], "to_geom": [],
                         "restriction": []}
    for d in restrictions:
        from_list = [(x["ref"], x["type"], get_geometry(x)) for x in d["members"] if x["role"] == "from"]
        via_list = [(x["ref"], x["type"], get_geometry(x)) for x in d["members"] if x["role"] == "via"]
        to_list = [(x["ref"], x["type"], get_geometry(x)) for x in d["members"] if x["role"] == "to"]
        if len(from_list) == 0:
            from_list = [(None,None,None)]
        if len(via_list) == 0:
            via_list = [(None,None,None)]
        if len(to_list) == 0:
            to_list = [(None,None,None)]
        total_len = len(from_list)*len(via_list)*len(to_list)
        for f in from_list:
            for v in via_list:
                for t in to_list:
                    restrictions_dict["from"].append(f[0])
                    restrictions_dict["via"].append(v[0])
                    restrictions_dict["to"].append(t[0])
                    restrictions_dict["from_type"].append(f[1])
                    restrictions_dict["via_type"].append(v[1])
                    restrictions_dict["to_type"].append(t[1])
                    restrictions_dict["from_geom"].append(f[2])
                    restrictions_dict["via_geom"].append(v[2])
                    restrictions_dict["to_geom"].append(t[2])
        restrictions_dict["type"].extend([d["type"]] * total_len)
        restrictions_dict["id"].extend([d["id"]] * total_len)
        restrictions_dict["restriction"].extend([d["tags"]["restriction"]] * total_len)
    restriction_df = pd.DataFrame(restrictions_dict)
    restriction_df = gpd.GeoDataFrame(restriction_df, geometry="via_geom")
    restriction_df = restriction_df[restriction_df["via_geom"].notnull()]
    if boundary:
        restriction_df = restriction_df[restriction_df["via_geom"].apply(lambda x: x.within(boundary))]
    return restriction_df

def simplify_boundary(geometry):
    if type(geometry)==shapely.geometry.multipolygon.MultiPolygon:
        lon_list, lat_list = zip(*[x for geo in geometry for x in geo.exterior.coords])
    else:
        lon_list, lat_list = zip(*geometry.exterior.coords)
    top, bottom = max(lat_list), min(lat_list)
    left, right = min(lon_list), max(lon_list)
    boundary = Polygon([[left, top], [left, bottom], [right, bottom], [right, top]])
    return boundary

def get_dist(node1, node2, node_dict_m):
    pt1 = node_dict_m[node1]
    pt2 = node_dict_m[node2]
    return math.sqrt(pow(pt1[0]-pt2[0], 2) + pow(pt1[1]-pt2[1], 2))

def find_centroid(group, signals):
    group = list(group)
    centroid = signals.iloc[group, [4,5]].mean()
    x, y = centroid
    return Point([x, y])


def find_adjacent(group, roads):
    return [ID for (ID, nodes) in zip(roads["id"], roads["nodes"]) for signal in list(group) if signal_dict[signal] in nodes]

def countPairs(theta_dict, Gi, f=lambda x: x < 20):
    a, b = zip(*theta_dict.items())
    n = len(a)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if f(abs(a[j] - a[i])):
                pairs.append((a[j], a[i]))
    return pairs

def dist(p1, p2):
    return math.sqrt(pow(p1[0]-p2[0], 2)+pow((p1[1]-p2[1]), 2))
