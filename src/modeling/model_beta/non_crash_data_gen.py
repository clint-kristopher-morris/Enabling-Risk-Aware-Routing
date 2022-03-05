import math
import random

import matplotlib.pyplot as plt
import pandas as pd

from src.constants import *


def load_statistical_traffic_disto(file_name):
    """ Load data about historic traffic seasonality daily and monthly
    """
    df = pd.read_csv(f'{DATA_PATH}/published_datasets/{file_name}')
    df.columns = ['dex', 'JAN', 'FEB', 'MAR', 'APR', 'MAY',
                  'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    df = df.set_index('dex')

    month_distri = df.sum(axis=0)
    for col in df.columns:
        sum_ = sum(df[col])
        df.loc[:, col] = df[col].map(lambda x: x / sum_)

    df["Ave"] = df.sum(axis=1) / 12
    month_distri.plot.bar(figsize=FIGSIZE, color=(0.8, 0, 0))
    plt.title('Traffic Monthly Distribution')
    plt.show()
    return df, month_distri


def plot_distribution(count_distribution, aadt_distribution, count_resampled,
                      title, s1='Segment Count', s2='AADT distribution', s3='Resampled Count'):
    """ Plot the distribution of resampled data
    Args:
        count_distribution (DataFrame): count of road segment types
        aadt_distribution (DataFrame): distribution of traffic among road type
        count_resampled (DataFrame): count of road segment types
        title (str): plot title
        s1 (str): sub title
        s2 (str): sub title
        s3 (str): sub title
    """
    index = ['Urban Interstate', 'Rural Interstate',
             'Urban Other Arterial', 'Rural Other Arterial',
             'Other Urban', 'Other Rural']
    data = pd.DataFrame()
    data.loc[:,s1] = count_distribution
    data.loc[:,s2] = aadt_distribution
    data.loc[:,s3] = count_resampled
    data.index = index
    _ = data.plot.bar(figsize=FIGSIZE, color=[(0.7, 0.7, 0.7), (0.8, 0, 0), (0.5, 0.5, 0.5)])
    plt.xlabel('Road Class', fontsize=13)
    plt.xticks(rotation=45)
    plt.ylabel('Percent of Total', fontsize=13)
    plt.title(title, fontsize=14)
    plt.show()


def sample_with_respect2aadt(road_info_sub, number_of_crashes):
    """ Sample data weighted by traffic data
    Args:
        road_info_sub (DataFrame): road data
        number_of_crashes (int): number of collisions
    Returns:
        resample_non_crash (DataFrame): resampled data
        distribution_aadt (array): distribution of AADT among class
    """
    # make one row for every AADT value and then sample accordingly.
    sum_aadt = int(sum(road_info_sub.AADT_DESGN))
    road_info_sub.loc[:, 'weights'] = road_info_sub['AADT_DESGN'].map(lambda x: int(x) / sum_aadt)
    resample_non_crash = road_info_sub.sample(n=number_of_crashes, replace=True, weights='weights')

    distribution_aadt = resample_non_crash['fClassSimp'].value_counts()[ROAD_CLASS_COLS]
    distribution_aadt = [x / sum(distribution_aadt) * 100 for x in distribution_aadt]
    return resample_non_crash, distribution_aadt


def add_month_column(resample_non_crash, month_distri, number_of_crashes):
    """ Sample month data weighted by traffic data
    Args:
        resample_non_crash (DataFrame): resampled non-crash data
        month_distri (array): weight of traffic per month
        number_of_crashes (int): number of collisions
    Returns:
        resample_non_crash (DataFrame): resampled data with month data
    """
    monthly_percent = month_distri / sum(month_distri)
    monthlyvalue = monthly_percent * number_of_crashes
    monthlyvalue.plot.bar(figsize=FIGSIZE, color=(0.8, 0, 0))
    plt.title('Resampled Monthly Distribution')
    plt.show()

    months = []
    for month, num_val in enumerate(monthlyvalue):
        months = months + [(month + 1) for x in range(math.ceil(num_val))]

    random.shuffle(months)
    months = months[:number_of_crashes]

    resample_non_crash.loc[:,'month'] = months
    resample_non_crash = resample_non_crash.reset_index()
    resample_non_crash = resample_non_crash.drop(['index'], axis=1)
    return resample_non_crash


def load_hourly_statistical_traffic_disto(file_name):
    """ Load hour/weekday data
    """
    traffic_df = pd.read_csv(f'{DATA_PATH}/published_datasets/{file_name}')
    total_traffic = traffic_df.iloc[:, 1:].values.sum()
    traffic_df = traffic_df.iloc[:, 1:] / total_traffic
    weekday2num = dict(zip(traffic_df.columns, [x for x in range(len(traffic_df.columns))]))
    return traffic_df, weekday2num


def fetch_hour_day_year(traffic_df, weekday2num, number_of_crashes):
    """ Sample hour/weekday data weighted by traffic data
    Args:
        traffic_df (DataFrame): distribution of daily and weekly data
        weekday2num (dict): map weekday to traffic fraction
        number_of_crashes (int): number of collisions
    Returns:
        resample_non_crash (DataFrame):
    """
    morph_df = {'value': [], 'weights': []}
    for idx, col in enumerate(traffic_df.columns):
        morph_df['value'] = morph_df['value'] + [f'{col}-{x}' for x in range(len(traffic_df))]
        morph_df['weights'] = morph_df['weights'] + traffic_df.iloc[:, idx].values.tolist()

    morph_df = pd.DataFrame(morph_df)
    morph_df.loc[:, 'day'] = morph_df.value.map(lambda x: weekday2num[x.split('-')[0]])
    morph_df.loc[:, 'hour'] = morph_df.value.map(lambda x: x.split('-')[1])

    morph_df = morph_df.sample(n=number_of_crashes, replace=True, weights='weights')
    morph_df = morph_df.reset_index()
    morph_df = morph_df.drop(['index', 'value'], axis=1)

    # generate year
    df_year = pd.DataFrame({'year': ['2020', '2018', '2019']})
    df_year = df_year.sample(n=number_of_crashes, replace=True)

    morph_df.loc[:, 'year'] = df_year['year'].values
    return morph_df


def array2percent(arr):
    return [x / sum(arr) * 100 for x in arr]


def main(df_crash, road_info, month_distri, df_traffic_by_hour, weekday2num):
    """ Generate non-crash data
    Args:
        df_crash (DataFrame): crash data
        road_info (DataFrame): road segment data
        month_distri (dict): monthly seasonality
        df_traffic_by_hour (DataFrame): hourly seasonality
        weekday2num (dict): weekly seasonality
    Returns:
        df_concat (DataFrame): correctly sampled non-crash data
    """
    road_info_sub = pd.DataFrame(road_info[['STR_UNQ_ID', 'AADT_DESGN', 'RU_F_SYSTE']])
    road_info_sub.loc[:, 'fClassSimp'] = road_info_sub['RU_F_SYSTE'].map(lambda x: FCLASS2STREET_MAP[x])

    count_distribution = array2percent(road_info_sub['fClassSimp'].value_counts()[ROAD_CLASS_COLS])
    # create a df that randomly selects the correct distrobtion of road segments
    number_of_crashes = len(df_crash)
    resample_non_crash, aadt_distribution = sample_with_respect2aadt(road_info_sub,
                                                                                        number_of_crashes)
    resample_non_crash = add_month_column(resample_non_crash, month_distri, number_of_crashes)
    count_resampled = array2percent(resample_non_crash.fClassSimp.value_counts()[ROAD_CLASS_COLS])
    plot_distribution(count_distribution, aadt_distribution, count_resampled,
                                         'Segment distributions')
    # get hourly data
    df_hour_day_year = fetch_hour_day_year(df_traffic_by_hour, weekday2num, number_of_crashes)
    # join time with selected segments
    df_concat = pd.concat([resample_non_crash, df_hour_day_year], axis=1)
    df_concat.to_csv(ROAD2WEATHER2DATE_PATH, index=False)  # save
    return df_concat
