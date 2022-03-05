import pandas as pd
from shapely import wkt

from src.constants import *


def nan_thresh_drop(df, thresh=10000):
    """ Drop columns with too many nan values.
    """
    nans = pd.DataFrame(df.isna().sum())
    nans = nans[nans[0] >= thresh]
    nan_cols = nans.index

    df = df.drop(nan_cols, axis=1)
    return df


def rush_hour(x):
    """ Sort numeric data into four bins.
    """
    if (int(x) >= 7) and (int(x) <= 10):
        x = 'AM_peak'
    elif (int(x) >= 15) and (int(x) <= 19):
        x = 'PM_peak'
    elif (int(x) > 10) and (int(x) < 15):
        x = 'Mid_day'
    else:
        x = 'Night/Early_Morning'
    return x


def cull_cols(df):
    """ Drop pre-determined useless data.
    """
    df = df.drop(USELESS_ROAD_COLS, axis=1)
    return df


def one_hot(new_name, old_name, df, exceptions=None):
    """ Custom one-hot encoding.
    """
    if exceptions:
        pass
    else:
        exceptions = ['NaN', np.nan]
    unique_lst = df[str(old_name)].unique()
    for item in unique_lst:
        if item in exceptions:
            df[str(new_name) + '_Other'] = np.where(df[str(old_name)] == item, 1, 0)
        else:
            df[str(new_name) + '_' + str(item)] = np.where(df[str(old_name)] == item, 1, 0)
    df = df.drop([str(old_name)], axis=1)
    return df


def dict_from_cols(df, col1, col2):
    return dict(zip(df[col1], df[col2]))


def reformat_time(x):
    hr, mins = x.split(':')
    mins, am_pm = mins.split(' ')
    hr, mins = int(hr), int(mins)
    if hr == 12:
        hr = hr - 12
    if am_pm == 'PM':
        hr = hr + 12
    return hr, mins  # mins from  12am


def get_crash_datetime(df, idx):
    """ Convert str to datetime values.
    """
    h, m = reformat_time(df.loc[idx, 'Crash_Time'])
    datetime_object = df.datetime.strptime(f"{df.loc[idx, 'Crash_Date']} {h}:{m}:00", '%m/%d/%Y %H:%M:%S')
    crash_id = df.loc[idx, 'Crash_ID']
    return datetime_object, crash_id


def rush_hour_peripheral(x):
    """ Sort numeric data into four bins.
    """
    if (int(x) >= 7) and (int(x) <= 10):
        x = 'Time_AM_peak'
    elif (int(x) >= 15) and (int(x) <= 19):
        x = 'Time_PM_peak'
    elif (int(x) > 10) and (int(x) < 15):
        x = 'Time_Mid_day'
    else:
        x = 'Time_Night/Early_Morning'

    timeData = {}
    for timeF in ['Time_PM_peak', 'Time_Mid_day', 'Time_AM_peak', 'Time_Night/Early_Morning']:
        if timeF == x:
            timeData[timeF] = 1
        else:
            timeData[timeF] = 0
    return timeData


def get_coords(x, val):
    """ GeoData preprocessing
    """
    try:
        x = x.coords[0][val]
    except:
        x = None
    return x


def preprocess_full_model_road(road_info):
    """ Preprocess road/segment data
    """
    for col in ['HSYS', 'RU_F_SYSTE', 'RU', 'MED_TYPE']:
        road_info = one_hot(col, col, road_info)

    road_info['geometry'] = road_info['geometry'].apply(wkt.loads)
    road_info.loc[:, 'lat'] = road_info['geometry'].map(lambda x: get_coords(x, 0))
    road_info.loc[:, 'long'] = road_info['geometry'].map(lambda x: get_coords(x, 1))

    lat_long2uni = {}
    longs, lats = road_info.long.tolist(), road_info.lat.tolist()
    for idx, x in enumerate(road_info.index.tolist()):
        lat_long2uni[x] = [longs[idx], lats[idx]]

    road_info_sub = road_info.drop(['STR_UNQ_ID', 'lat', 'long', 'geometry'], axis=1)
    segment_map = road_info_sub.to_dict('index')
    return segment_map, lat_long2uni
