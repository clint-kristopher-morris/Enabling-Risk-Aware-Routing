import pandas as pd
from eli5.sklearn import PermutationImportance

from src import ups_data_loader, preprocessing, ups_plotting
from src.constants import *
from src.modeling import boosted_modeling


def load_preprocess(file_c, file_nc):
    """
    Loads and preprocess the two datasets.

    Args:
        file_c (str): file name for crash data
        file_nc (str): file name for non-crash data

    Returns:
        df_c (DataFrame): dataframe of collision data
        df_nc (DataFrame): dataframe of non-crash data
    """
    # load data
    df_c = pd.read_csv(file_c)
    df_nc = pd.read_csv(file_nc)
    # convert days2nums
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday', 'Saturday']
    num = [x for x in range(len(days))]
    day2num = dict(zip(num, days))
    # time encode
    df_nc.loc[:, 'Time'] = df_nc['hour'].map(lambda x: preprocessing.rush_hour(int(x)))
    df_nc.loc[:, 'Day'] = df_nc['day'].map(lambda x: day2num[int(x)])
    df_nc = preprocessing.one_hot('Day', 'Day', df_nc)
    df_nc = preprocessing.one_hot('Time', 'Time', df_nc)
    return df_c, df_nc


def check_for_missing(cols1, cols2):
    """ Check from non-intersecting columns.
    """
    missing_cols = []
    for col in cols1:
        if col not in cols2:
            missing_cols.append(col)
    return missing_cols


def drop_check_for_missing(df_c, df_nc):
    """
    checks for columns that do not intersect both dataframes
    and drops columns that are not used.

    Args:
        df_c (DataFrame): dataframe of collision data
        df_nc (DataFrame): dataframe of non-crash data

    Returns:
        df_c (DataFrame):
        df_nc (DataFrame):
    """
    # drop cols
    c_drops = ['Crash_ID', 'Crash_Sev_ID', 'Tot_Injry_Cnt', 'target', 'SCHOOL_ZN']
    nc_drops = ['month', 'day', 'hour', 'year', 'wc', ]
    df_c, df_nc = df_c.drop(c_drops, axis=1), df_nc.drop(nc_drops, axis=1)
    # check missing
    missing_cols_c = check_for_missing(df_c.columns, df_nc.columns)
    missing_cols_nc = check_for_missing(df_nc.columns, df_c.columns)
    count_of_missing_cols = len(missing_cols_c + missing_cols_nc)
    print(f'Count of mismatching columns: {count_of_missing_cols}')
    return df_c, df_nc


def clean_preprocess_fullset(df_full, one_hot_cols, keep_cols):
    """ Preprocessing data.
    """
    for col in one_hot_cols:
        df_full = preprocessing.one_hot(col, col, df_full)

    df_full = df_full[keep_cols]
    df_full = df_full.dropna()
    return df_full


def main(df_C, df_NC, label='target', split_frac=0.2, model_name='Model_Alpha'):
    """ Conducts alpha model training (model that predicts crash occurrence).
    Args:
        df_C (DataFrame): crash data
        df_NC (DataFrame): non-crash data
        label (str): name of label column
        split_frac (str): test train split fraction
        model_name (str): model name used for saving model
    Returns:
        perm (PermutationImportance): permutation importance of features
        cols (list): list of columns used
        x_valid (DataFrame): validation predictor variables
        y_valid (DataFrame): validation labels
    """
    # add labels
    df_C[label], df_NC[label] = [1] * len(df_C), [0] * len(df_NC)
    # combine into a single frame
    df_full = pd.concat([df_C, df_NC])
    # preprocess the full set
    one_hot_cols = ['RU_F_SYSTE', 'RU', 'MED_TYPE']
    df_full = clean_preprocess_fullset(df_full, one_hot_cols, BST_COLS_ALPHA_MODEL + [label])
    # plot heat map of features
    ups_plotting.plot_heat_map(df_full.drop([label], axis=1), figsize=8, fontsize=12)
    # train model
    x_train, x_valid, y_train, y_valid = boosted_modeling.split_data(df_full, label, split_frac)
    model_alpha = boosted_modeling.train_catboost(MODEL_ALPHA_PARAS, x_train, x_valid, y_train, y_valid)
    ups_data_loader.pickel_model(model_name, model_alpha, x_train.columns)
    # get permutation importance
    perm = PermutationImportance(model_alpha).fit(x_valid, y_valid)
    # plot SHAP
    _, _ = ups_plotting.plot_shap(x_train, model_alpha)
    return perm, x_train.columns.tolist(), x_valid, y_valid
