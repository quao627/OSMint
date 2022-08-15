import pickle
import networkx as nx
from functions import *

def get_data(city, state):
    boundary = ox.geocode_to_gdf(f'{city}, {state}, USA')
    boundary = list(boundary["geometry"])[0]
    if type(boundary) != shapely.geometry.polygon.Polygon:
        boundary = boundary[0]
    polygon = simplify_boundary(boundary)

    # buffer the polygon for 1000m to get complete road network

    print("Collecting roads...")
    buffer_dist = 1000
    poly_proj, crs_utm = projection.project_geometry(boundary)
    poly_proj_buff = poly_proj.buffer(buffer_dist)
    poly_buff, _ = projection.project_geometry(poly_proj_buff, crs=crs_utm, to_latlong=True)
    network_type = "drive"
    custom_filter = None
    roads = ox.downloader._osm_network_download(poly_buff, network_type, custom_filter)

    nodes = [road for road in roads[0]["elements"] if road["type"] == "node"]
    roads = [road for road in roads[0]["elements"] if road["type"] == "way"]

    node_dict = {node['id']: [node['lon'], node['lat']] for node in nodes}
    node_dict_m = gpd.GeoDataFrame(index=node_dict.keys(), geometry=[Point(p) for p in node_dict.values()]).set_crs(
        {'init': 'epsg:4326'}).to_crs({'init': 'epsg:3395'})
    node_dict_m = {index: [row["geometry"].xy[0][0], row["geometry"].xy[1][0]] for index, row in node_dict_m.iterrows()}
    geometry = [LineString([node_dict[node] for node in road["nodes"]]) for road in roads]
    for road in roads:
        road.update(road.pop("tags"))
    useful_tags = ['name', 'type', 'id', 'nodes', 'highway', 'lanes', 'lanes:backward',
                   'lanes:both_ways', 'lanes:forward',
                   'oneway', 'turn:lanes', 'turn:lanes:backward', 'turn:lanes:both_ways',
                   'turn:lanes:forward', 'width', 'maxspeed', 'incline']
    roads = pd.DataFrame(roads)
    roads = roads[[tag for tag in roads.columns if tag in useful_tags]]
    roads = gpd.GeoDataFrame(roads, geometry=geometry)

    roads = roads.set_crs({'init': 'epsg:4326'})
    roads = roads.to_crs({'init': 'epsg:3395'})
    # roads = roads[~roads["highway"].isin(["motorway","motorway_link"])].reset_index().drop("index", axis=1)

    # build a graph containing all the nodes and their connections

    print("Constructing graph...")
    G = nx.MultiGraph()
    nodes = set()
    paths = set()
    # edge_road = defaultdict(set)
    for index, row in roads.iterrows():
        path_nodes = [group[0] for group in itertools.groupby(row["nodes"])]
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(
            [(path_nodes[i], path_nodes[i + 1], get_dist(path_nodes[i], path_nodes[i + 1], node_dict_m)) for i in
             range(len(path_nodes) - 1)], weight="length")

    # edge_geo = {tuple(sorted([u, v])): LineString([node_dict[u], node_dict[v]]) for (u,v) in G.edges()}
    # edge_node = {tuple(sorted([u, v])): set([u,v]) for (u,v) in G.edges()}

    for u, v, k in G.edges(keys=True):
        G[u][v][k]["nodes"] = set([u, v])
        G[u][v][k]["geometry"] = LineString([node_dict[u], node_dict[v]])

    # remove all nodes with degree equal to 2
    while any([degree == 2 and len(G[node]) > 1 for node, degree in G.degree()]):
        bi_nodes = [node for node, degree in G.degree() if degree == 2]
        for index, node in enumerate(bi_nodes):
            if node in G.nodes():
                neighbors = list(dict(G[node]).keys())
                if len(neighbors) == 2:
                    if neighbors[0] == neighbors[1]:
                        continue
                    l1, l2 = sum([G[node][neighbors[0]][k]["length"] for k in dict(G[node][neighbors[0]]).keys()]), sum(
                        [G[node][neighbors[1]][k]["length"] for k in dict(G[node][neighbors[1]]).keys()])
                    combined_nodes = [G[node][neighbors[0]][k]["nodes"] for k in dict(G[node][neighbors[0]]).keys()] + [
                        G[node][neighbors[1]][k]["nodes"] for k in dict(G[node][neighbors[1]]).keys()]
                    combined_nodes = [node for nodes in combined_nodes for node in nodes]
                    k = G.add_edge(*neighbors)
                    G[neighbors[0]][neighbors[1]][k]["length"] = l1 + l2
                    G[neighbors[0]][neighbors[1]][k]["nodes"] = set(combined_nodes)
                    G[neighbors[0]][neighbors[1]][k]["geometry"] = shapely.ops.linemerge(
                        [G[node][neighbors[0]][k]["geometry"] for k in dict(G[node][neighbors[0]]).keys()] + [
                            G[node][neighbors[1]][k]["geometry"] for k in dict(G[node][neighbors[1]]).keys()])
                    G.remove_node(node)
    G.remove_edges_from(nx.selfloop_edges(G))

    # get traffic signals

    print("Collecting traffic signals...")
    sleep(10)
    signals = get_traffic_signals(polygon=polygon, boundary=boundary)
    signals['keep'] = signals["id"].isin(node_dict)

    nodes = gpd.GeoDataFrame(geometry=[Point(node_dict[node]) for node in G.nodes()], index=G.nodes())
    nodes = nodes.set_crs({'init': 'epsg:4326'})
    nodes = nodes.to_crs({'init': 'epsg:3395'})
    nodes['x'] = nodes['geometry'].apply(lambda x: x.xy[0][0])
    nodes['y'] = nodes['geometry'].apply(lambda x: x.xy[1][0])

    signals = signals[signals["keep"]]
    signals = signals.reset_index().drop("index", axis=1)
    print("Finished collecting signals")
    groups = find_clusters(signals, threshold=80)
    print(f"{len(groups)} signalized intersections are found.")

    signal_dict = dict(zip(signals.index, signals["id"]))

    signal_centroids = [find_centroid(group, signals) for group in groups]
    df_group = pd.DataFrame({"group": groups, "centroids": signal_centroids})
    df_group = gpd.GeoDataFrame(df_group, geometry="centroids").set_crs({"init": "epsg:3395"})

    # get adjacent roads
    print("Matching adjacent roads...")
    adjacent_roads = []
    for index, group in enumerate(df_group["group"]):
        adjacent_roads.append(tuple(find_adjacent(group, roads)))
        print(index, end="\r")

    df_group["adjacent_roads"] = adjacent_roads
    df_group["nodes"] = df_group["group"].apply(lambda x: [signal_dict[index] for index in list(x)])

    # get restrictions
    print("Collecting turn restrictions...")
    sleep(10)
    restrictions = get_turn_restrictions(polygon=polygon, boundary=boundary)
    restrictions = restrictions.dropna().reset_index().drop("index", axis=1)

    return {"roads": roads, "nodes": nodes, "node_dict": node_dict,
                     "node_dict_m": node_dict_m,
                     "G": G,
                     "signals": signals,
                     "df_group": df_group,
                     "restrictions": restrictions}
