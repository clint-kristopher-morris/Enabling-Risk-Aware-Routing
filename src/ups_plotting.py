import warnings

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
import matplotlib as mpl
from svgpathtools import svg2paths
from svgpath2mpl import parse_path

warnings.filterwarnings("ignore")

from src.constants import *


def plot_heat_map(df, figsize=48, fontsize=22):
    """Plots heat map give a DataFrame.
    """
    labels = df.columns.tolist()
    # --- Correlation Matrix Heatmap --- #
    f, ax = plt.subplots(figsize=(figsize, figsize))
    corr = df.corr()
    _ = sns.heatmap(round(corr, 4), ax=ax, cmap="coolwarm", fmt='.2f', linewidths=1, annot_kws={"size": 10})
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    plt.show()


def plot_shap(x_train, model, n=40000):
    """Generate and plot SHAP: Shapley Additive Explanations.
    """
    explainer = shap.TreeExplainer(model)
    sample = x_train.sample(n=n)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample)
    plt.show()
    return shap_values, sample


def pd2gpd(gpd_df):
    """Converts DataFrame to GeoDataFrame.
    """
    gpd_df = gpd.GeoDataFrame(gpd_df, geometry=gpd.points_from_xy(gpd_df.Longitude, gpd_df.Latitude),
                              crs={'init': f'EPSG:{LAT_LON_EPSG}'})
    return gpd_df


def plot_point_map(gpd_df, ax, title, color, markersize=0.05, type_='t1'):
    """Handles all Geopandas map plotting.
    """
    map_background = cx.providers.Stamen.TonerLite
    gpd_df = gpd_df.to_crs(epsg=WEB_MERCATOR_EPSG)
    gpd_df.reset_index(inplace=True, drop=True)
    if type_ == 't1':
        gpd_df.plot(figsize=(10, 10), markersize=markersize, alpha=0.3, ax=ax, color=color)
    elif type_ == 'weather_stations':
        _, icon_attributes = svg2paths(WEATHER_STATION_ICON_PATH)
        station_icon = parse_path(icon_attributes[0]['d'])
        station_icon.vertices -= station_icon.vertices.mean(axis=0)
        station_icon = station_icon.transformed(mpl.transforms.Affine2D().rotate_deg(180))
        station_icon = station_icon.transformed(mpl.transforms.Affine2D().scale(-1, 1))
        gpd_df.plot(figsize=(10, 10), marker=station_icon, markersize=markersize, alpha=0.7, ax=ax, color=color)
    else:
        gpd_df.plot(figsize=(10, 10), ax=ax, color=color)
    ax.set_title(title)
    cx.add_basemap(ax, source=map_background)


def plot_weather_stations():
    """Plotting sample of weather station density.
    """
    weather_data_gpd = pd.read_csv(WEATHER_LOCATIONS_PATH)
    weather_data_gpd = weather_data_gpd[weather_data_gpd.State==TARGET_STATE]
    weather_data_gpd = pd2gpd(weather_data_gpd)

    dallas_tl = [32.948107, -97.082292]
    dallas_br = [32.630157, -96.482470]

    dallas_gdf = weather_data_gpd[(weather_data_gpd.Latitude > dallas_br[0])]
    dallas_gdf = dallas_gdf[(dallas_gdf.Latitude < dallas_tl[0])]
    dallas_gdf = dallas_gdf[(dallas_gdf.Longitude > dallas_tl[1])]
    dallas_gdf = dallas_gdf[(dallas_gdf.Longitude < dallas_br[1])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plot_point_map(weather_data_gpd, ax1, 'Weather Stations', T_RED, markersize=15, type_='weather_stations')
    plot_point_map(dallas_gdf, ax2, 'Dallas Weather Stations', T_RED, markersize=1200, type_='weather_stations')


def plot_c_cn_points(df_c, df_nc, figsize=(20, 10)):
    """Geopandas plot all crash and non-crash maps.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    # plot crash data
    df_c = pd2gpd(df_c)
    _ = plot_point_map(df_c, ax1, 'All Crashes in Texas 2017-2020', T_GRAY)
    # plot non-crash data
    df_nc = pd2gpd(df_nc)
    _ = plot_point_map(df_nc, ax2, 'All Generated Non-Crash Data 2017-2020', T_RED)
    plt.show()


def plot_balanced_data(df_balanced, df):
    """Plot resampled data distribution.
    """
    plot_data = pd.DataFrame()
    plot_data['Resampled Crash Data'] = df_balanced['target'].value_counts()
    plot_data['Crash Data'] = df['target'].value_counts()
    index_ = [f'Class {cl}' for cl in df['target'].value_counts().index]
    plot_data.index = index_
    plot_data.plot.bar(figsize=FIGSIZE, color=[T_GRAY, T_RED])
    plt.ylabel('Count of Collision Severity')
    plt.xlabel('Collision Severity')
    plt.xticks(rotation=0)
    plt.show()
