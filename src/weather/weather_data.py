import calendar, random
from shapely import wkt
import geopandas as gpd
import dload
import datetime
import time
import pandas as pd

from src.constants import *
from src.common_tools import load_obj, save_obj


def randomdate(year, month, weekday_val):
    """ Given a year, month and day of the week generate a date
    Args:
        year (int): e.g. 2017
        month (int): month value. range 0-11
        weekday_val (int): weekday value. range 0-5

    Returns:
        date (str): %m/%d/%Y
    """
    dates = calendar.Calendar().itermonthdates(year, month)
    return random.choice([date for date in dates if (date.month == month and date.weekday() == weekday_val)])


def get_weather_data(lat, long, date='03/01/2020'):
    """ Works with Weather.com API
    Args:
        lat (int): latitude
        long (int): longitude
        date (str): %m/%d/%Y

    Returns:
        weather data (dict):
    """
    date = datetime.datetime.strptime(date, '%m/%d/%Y')
    date = date.strftime('%Y%m%d')
    url = f'https://api.weather.com/v1/geocode/{lat}/{long}/observations/historical.json?apiKey={API_KEY}startDate={date}&endDate={date}'
    return dload.json(url)


def collect_raw_weather_data(df, start_val=0, stid2lat=None, stid2long=None):
    """ Collect and save's raw weather that is matched to a date and location of an incident (crash/non-crash).
    Args:
        df (DataFrame): df containing time and location information of all incidents.
        start_val (int): value used if you want to start from checkpoint.
        stid2lat (str): dict mapping incident to latitude
        stid2long (str): dict mapping incident to longitude
    """
    weather_data, date_data = {}, {}
    print('start')
    s = time.time()
    for idx, row in df.iterrows():
        if idx <= start_val:
            continue
        if idx % 1000 == 0:
            print(idx)
        if idx % COLLECTION_SAVE_THRESH == 0 and idx != 0:
            save_obj(weather_data, f'{PATH2WEATHER_OBJ_DIR}/data/weather_data-{idx}')
            save_obj(date_data, f'{PATH2WEATHER_OBJ_DIR}/meta/date_data-{idx}')
            print(f'loop time: {time.time() - s}')
            s = time.time()
            weather_data, date_data = {}, {}

        random_gen_date = randomdate(row.year, row.month, row.day)
        random_gen_date = random_gen_date.strftime('%m/%d/%Y')
        data = get_weather_data(
            stid2lat[row.STR_UNQ_ID],
            stid2long[row.STR_UNQ_ID],
            date=random_gen_date)

        weather_data[row.NonCrashIDX] = data
        date_data[row.NonCrashIDX] = random_gen_date


def geo_convert(df):
    df['geometry'] = df['geometry'].apply(wkt.loads)
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df['centroid'] = df.geometry.map(lambda x: x.centroid)
    df['Latitude'] = df.centroid.map(lambda x: x.xy[1][0])
    df['Longitude'] = df.centroid.map(lambda x: x.xy[0][0])
    return df


def save_weather_obj_files(tx_crash, start_number=0, update_iter=1000):
    """ Preprocesses raw data.
    Args:
        tx_crash (DataFrame):
        start_number (int): checkpoint
        update_iter (int): interval of update print out
    """
    weather_data = {}
    s = time.time()
    for f, idx in enumerate(range(start_number, len(tx_crash))):
        if idx % update_iter == 0:
            print(idx)
        if idx % COLLECTION_SAVE_THRESH == 0 and f != 0:
            save_obj(weather_data, f'weather_data-{idx}')
            print(f'loop time: {time.time() - s}')
            s = time.time()
            weather_data = {}

        data = get_weather_data(tx_crash.loc[idx, 'Latitude'], tx_crash.loc[idx, 'Longitude'],
                                date=tx_crash.loc[idx, 'Crash_Date'])
        weather_data[tx_crash.loc[idx, 'Crash_ID']] = data
    return weather_data


def convert_raw_weather_obj2csv(tx_crash, dicts, csvs, update_iter=1000):
    """ Loads raw weather data and formats it for modeling. Also matches incident to exact times
    Args:
        tx_crash (DataFrame):
        dicts (list): list of all saved raw weather data
        csvs (list): list of all processed weather data
        update_iter (int): interval of update print out
    """
    for dict_name in dicts:
        if dict_name.replace('.pkl','.csv') in csvs:
            print(f'{dict_name} has already been processed')
            continue

        print(f'processing: {dict_name} ...')
        output = pd.DataFrame() # init dataframe

        print(dict_name) # load weather dict
        number = int(dict_name.split('-')[-1].split('.')[0])
        weather_dict = load_obj(f'{PATH2WEATHER_OBJ_DIR}/data/weather_data-{number}')

        for crash_idx in range((number-COLLECTION_SAVE_THRESH),number):
            if crash_idx%update_iter == 0:
                print(crash_idx)

            crash_id = tx_crash.loc[crash_idx,'Crash_ID']
            date_of_crash, time_of_crash = tx_crash.loc[crash_idx,'Crash_Date'], tx_crash.loc[crash_idx,'Crash_Time']
            crash_time = datetime.datetime.strptime(f'{date_of_crash} {time_of_crash}', '%m/%d/%Y %I:%M %p')
            # loop vals for weather loop
            min_time = np.inf
            best_weather_idx = 0
            try:
                weather_datalist = weather_dict[crash_id]['observations']
            except KeyError:
                # no 'observations' append crash ID and nan's to df
                nan_dict = {'Crash_ID':crash_id}
                output = output.append(nan_dict, ignore_index=True)
                continue

            for idx, data in enumerate(weather_datalist):
                weather_epoch = data['valid_time_gmt']
                weather_time = datetime.datetime.fromtimestamp(weather_epoch)
                dif = (crash_time-weather_time).total_seconds()
                if abs(dif) < min_time:
                    min_time = dif
                    best_weather_idx = idx

            # add matching crash ID
            weather_datalist[best_weather_idx]['Crash_ID'] = crash_id
            # append to df
            output = output.append(weather_datalist[best_weather_idx], ignore_index=True)
        # save csv
        output.to_csv(f'{PATH2OUTPUT_CSV_NC_DIR}/weather_data-{number}.csv')

