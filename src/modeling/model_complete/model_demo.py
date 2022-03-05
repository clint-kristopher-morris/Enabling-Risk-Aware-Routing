import random

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from src.constants import *
from src.modeling.model_complete import model_full
from src import ups_plotting

def fetch_map_demo_data():
    """ Loads a nx network map of Fredericksburg TX for model demonstration.
    """
    road_info = gpd.read_file(MAP_DATA_PATH)
    road_info['STR_UNQ_ID'] = road_info.index

    fredericksburg = road_info[road_info.CITY == FREDERICKSBURG_ID]
    f, ax = plt.subplots(figsize=(10,10))
    ups_plotting.plot_point_map(fredericksburg, ax, 'Fredericksburg TX', T_RED, type_='t1')
    plt.show()
    return fredericksburg


def load_plot_graph():
    """ Plots the nx network graph.
    """
    g = ox.graph_from_place(DEMO_LOCATION, network_type='drive')
    g_projected = ox.project_graph(g)
    ox.plot_graph(g_projected)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(g)
    return g, gdf_nodes, gdf_edges


def ckdnearest(gd_a, gd_b):
    """ Map all open map nx network id's to our shape file
    Args:
        gd_a (GeoDataFrame): gpd shape file
        gb_b (GeoDataFrame): gpd nx network
    Returns:
        gdf (GeoDataFrame): map to match segments
    """
    n_a = np.array(list(gd_a.geometry.apply(lambda x: (x.x, x.y))))
    n_b = np.array(list(gd_b.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(n_b)
    dist, idx = btree.query(n_a, k=1)
    gd_b_nearest = gd_b.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gd_a.reset_index(drop=True),
            gd_b_nearest,
            pd.Series(dist, name='dist')
        ],
        axis=1)
    return gdf


def map_street_id2edges(segments, gdf_edges):
    """ Map all open map nx network id's to our shape file.
    Args:
        segments (GeoDataFrame): gpd shape file
        gdf_edges (GeoDataFrame): nx network
    Returns:
        gdf (GeoDataFrame): map to match segments
    """
    nx_ = pd.DataFrame()
    nx_['centroid'] = gdf_edges.geometry.centroid
    nx_['osmid'] = gdf_edges.osmid
    nx_ = nx_.reset_index()
    nx_gdf = gpd.GeoDataFrame(nx_, geometry=nx_.centroid)

    segments_sub = pd.DataFrame()
    segments_sub['centroid'] = segments.geometry.centroid
    segments_sub['STR_UNQ_ID'] = segments.STR_UNQ_ID
    segments_sub = segments_sub.reset_index()
    segments_sub_gdf = gpd.GeoDataFrame(segments_sub, geometry=segments_sub.centroid)

    map_df = ckdnearest(nx_gdf, segments_sub_gdf)
    return map_df


def get_route_in_unq(route, map_df):
    """
    Args:
        route (list): list of nx edges in the selected trip
        map_df (DataFrame): map nx to segment id's
    Returns:
        str_urq_ids (list): segment id's in the trip
    """
    str_urq_ids = []
    for idx in range(len(route) - 1):
        str_urq_ids.append(map_df[map_df.u == route[idx]][map_df.v==route[idx + 1]].STR_UNQ_ID.values[0])
    return str_urq_ids


def random_map(g, map_df, nd, model_alpha, model_beta, lat_long2uni, segment_map):
    """ Map all open map nx network id's to our shape file.
    Args:
        g (NxNetwork): full network
        map_df (DataFrame): map of nx to segment id's with extended info for modeling
        nd (list): list of network nodes
        model_alpha (CatBoost Model): calculates probability of collision
        model_beta (CatBoost Model): given a collision occurs, this model calculates probability of each crash severity.
        lat_long2uni (dict): coordinates to id of segment
        segment_map (dict): map segments to data
    """
    route = nx.shortest_path(g, nd[random.randint(0, len(nd))], nd[random.randint(0, len(nd))])
    str_urq_ids = get_route_in_unq(route, map_df)
    # compute
    cost = model_full.fetch_trip_cost(model_alpha, model_beta, lat_long2uni, segment_map, str_urq_ids)
    print(f'This route incurs a risk cost of: ${round(cost, 2)}')
    # plot
    fig, ax = ox.plot_graph_route(g, route, route_linewidth=6, node_size=0, bgcolor='k', show=False, close=False)
    fig.text(0.15, 0.81, f'Incurred Risk Cost: ${round(cost, 2)}', fontsize=12, color='white', fontweight='bold')
    plt.show()

def fetch_demo_maps(target_city_streets):
    """ Load network map information for demonstration.
    """
    g, gdf_nodes, gdf_edges = load_plot_graph()
    map_df = map_street_id2edges(target_city_streets, gdf_edges)
    nd = list(g.nodes)
    return g, map_df, nd
