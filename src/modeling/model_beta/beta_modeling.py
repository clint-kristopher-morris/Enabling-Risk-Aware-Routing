import pandas as pd
from eli5.sklearn import PermutationImportance

from src.constants import *
from src import preprocessing
from src.modeling import boosted_modeling
from src import ups_data_loader
from src import ups_plotting


def mutual_exclusive_helper(target, dict_, ordered_cols):
    """ Helper to the mutual_exclusive func. Make collision severity mutually exclusive.
    Args:
        target (int): crash severity
        dict_ (dict): map of crash severity
        ordered_cols (list): crash severity in ascending order of severity
    Returns:
        dict_ (dict): mutually exclusive map
    """
    newlist = ordered_cols.copy()
    newlist.remove(target)
    for idx in dict_[target]:
        target_val = dict_[target][idx]
        if target_val == 1:
            for col in newlist:
                dict_[col][idx] = 0
    return dict_


def mutual_exclusive(df_injry):
    """ Convert multi-column set in injury severity into a single target column.
    Args:
        df_injry (DataFrame): crash severity
    Returns:
        df_clean_injry (DataFrame): single target column
    """
    for col in df_injry.columns:
        df_injry[col] = df_injry[col].map(lambda x: 0 if x == 0 else 1)
    # order of importance
    ordered_cols = ['Death_Cnt', 'Incap_Injry_Cnt', 'Nonincap_Injry_Cnt',
                    'Unkn_Injry_Cnt', 'Poss_Injry_Cnt','Non_Injry_Cnt']
    dict_ = df_injry.to_dict(orient='dict')
    for target_col in ordered_cols:  # cycle
        dict_ = mutual_exclusive_helper(target_col, dict_, ordered_cols)

    df_clean_injry = pd.DataFrame(dict_)  # 2df
    for idx, col in enumerate(df_clean_injry.columns):
        idx = idx + 1
        df_clean_injry[col] = df_clean_injry[col].map(lambda x: idx if x == 1 else 0)
    # get target columns
    df_clean_injry["target"] = df_clean_injry.sum(axis=1)
    return df_clean_injry


def split_injury_and_label_crash_data(df_crash):
    """ Clean data and split predictor variables from label data.
    """
    df_clean_crash = pd.DataFrame(df_crash)  # copy crash data
    df_clean_crash = df_clean_crash.apply(pd.to_numeric)  # fix format

    df_injry = df_clean_crash[INJRY_COLS]  # sub-select
    df_clean_crash = df_clean_crash.drop(INJRY_COLS, axis=1)
    df_injry = df_injry.rename(columns={'Sus_Serious_Injry_Cnt': 'Incap_Injry_Cnt'})
    df_injry = mutual_exclusive(df_injry)

    df_clean_crash['target'] = df_injry['target']
    return df_clean_crash, df_injry


def balance_targets(df):
    """ Balance the data by crash severity under-sample major classes and
    over sample minor classes.
    """
    mean_count = int(df['target'].value_counts().mean())
    df_balanced = pd.DataFrame()
    for val in df['target'].value_counts().index:
        addition = df[df['target'] == val].sample(n=mean_count, replace=True)
        df_balanced = df_balanced.append(addition)
    ups_plotting.plot_balanced_data(df_balanced, df)
    return df_balanced


def segment_data_preprocessing(df_crash, road_info):
    """ Preprocess road/segment data.
    """
    df_clean_crash, df_injry = split_injury_and_label_crash_data(df_crash)
    # match on street labels
    df_joined = df_clean_crash.merge(road_info, how='left', on='STR_UNQ_ID')
    # one hot cat cols
    for col in ['HSYS', 'RU_F_SYSTE', 'RU', 'MED_TYPE']:
        df_joined = preprocessing.one_hot(col, col, df_joined)

    # remove class zero it is an error
    df_cleaned = df_joined[df_joined.target != 0]
    df_cleaned = df_cleaned.dropna()
    return df_cleaned


def main(df_cleaned, label='target', split_frac=0.2, model_name='Model_Beta'):
    """ Conducts beta model training (model that predicts crash severity).
    Args:
        df_cleaned (DataFrame): preprocess data for beta modeling
        label (str): name of label column
        split_frac (str): test train split fraction
        model_name (str): model name used for saving model
    Returns:
        perm (PermutationImportance): permutation importance of features
        cols (list): list of columns used
        x_valid (DataFrame): validation predictor variables
        y_valid (DataFrame): validation labels
    """
    df_cleaned = df_cleaned[[label] + BST_COLS_BETA_MODEL]
    ups_plotting.plot_heat_map(df_cleaned.drop([label], axis=1), figsize=8, fontsize=12)

    x_train, x_valid, y_train, y_valid = boosted_modeling.split_data(df_cleaned, label, split_frac)
    # add target to x_train for class balancing
    x_train[label] = y_train.values
    x_train_balance = balance_targets(x_train)
    y_train = x_train_balance[label]
    x_train = x_train_balance.drop([label], axis=1)

    model_beta = boosted_modeling.train_catboost(MODEL_BETA_PARAMS, x_train, x_valid, y_train, y_valid)

    ups_data_loader.pickel_model(model_name, model_beta, x_train.columns)
    # get permutation importance
    perm = PermutationImportance(model_beta).fit(x_valid, y_valid)
    # plot SHAP
    _, _ = ups_plotting.plot_shap(x_train, model_beta)
    return perm, x_train.columns.tolist(), x_valid, y_valid