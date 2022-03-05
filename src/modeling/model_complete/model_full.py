import pandas as pd
from src.constants import *
import datetime
from src import preprocessing
from src.weather.weather_data import get_weather_data


def get_date_data(day_val):
    """Format weekday data.
    """
    day_val = NUM2DAY[day_val]
    day_data = {}
    for day in DAYS:
        if day == day_val:
            day_data[day] = 1
        else:
            day_data[day] = 0
    return day_data


def get_data_from_segment(road_id, lat_long2uni, segment_map):
    """Given a date and a coordinates collect weather data from that day.
        Can be used live to collect weather data of trip path.
    Args:
        road_id (int): id of the segment
        lat_long2uni (dict): coordinates to id of segment
        segment_map (dict): map segments to data
    Returns:
        all_data (dict): weather data mapped to segment
    """
    lat, long = lat_long2uni[road_id]
    data = segment_map[road_id]
    # init time
    date_now = datetime.datetime.now()
    day_val, h = date_now.weekday(), int(date_now.strftime('%H'))
    # get temporal data
    day_data = get_date_data(day_val)
    time_data = preprocessing.rush_hour_peripheral(h)
    # get weather
    weather_data = get_weather_data(lat, long, date_now.strftime('%m/%d/%Y'))['observations'][-1]
    weather_data_now = {}
    for col in ['precip_hrly', 'vis', 'wspd']:
        weather_data_now[col] = weather_data[col]

    all_data = {}
    for d in [data, day_data, time_data, weather_data_now]:
        all_data.update(d)
    return all_data


def fetch_trip_cost(model_alpha, model_beta, lat_long2uni, segment_map, segments):
    """Implement both model alpha and beta to calculate risk cost per segment.
    Args:
        model_alpha (CatBoost Model): calculates probability of collision
        model_beta (CatBoost Model): given a collision occurs, this model calculates probability of each crash severity.
        lat_long2uni (dict): coordinates to id of segment
        segment_map (dict): map segments to data
        segments (list): list of segments in the given trip
    Returns:
        cost (float): risk cost
    """
    data = {}
    for idx, road_id in enumerate(segments):
        data[idx] = get_data_from_segment(road_id, lat_long2uni, segment_map)
    data = pd.DataFrame(data).T

    prob_nc_c = np.sum(model_alpha.predict_proba(data[BST_COLS_ALPHA_MODEL]), axis=0)[
                    1] * DISTRO_CRASH / 50  # procrash*likelyHood
    prob_severity = np.sum(model_beta.predict_proba(data[BST_COLS_BETA_MODEL]), axis=0) * DISTRO_SEVER
    cost = prob_nc_c * sum(prob_severity * COST_ARR)
    return cost
