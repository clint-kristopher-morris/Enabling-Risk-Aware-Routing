import pandas as pd

from src.constants import *


def load_all_tx_crash_data():
    """ Load raw collision data from TxDOT.
    """
    tx_crash1 = pd.read_csv(f'{PATH2CRASH_DIR}/2020 crash.csv', low_memory=False)
    tx_crash2 = pd.read_csv(f'{PATH2CRASH_DIR}/2019 crash.csv', low_memory=False)
    tx_crash3 = pd.read_csv(f'{PATH2CRASH_DIR}/2018 crash.csv', low_memory=False)
    tx_crash = pd.concat([tx_crash1, tx_crash2, tx_crash3])
    tx_crash = tx_crash[tx_crash['Latitude'].notna()]
    return tx_crash


def load_crash_and_weather(file_name):
    """ Load collision data.
    """
    df = pd.read_csv(f'{DATA_PATH}/{file_name}')
    keep_cols = ['Crash_ID', 'precip_hrly', 'vis', 'wc', 'wspd', 'Crash_Speed_Limit',
                 'Road_Cls_ID', 'Crash_Sev_ID', 'Sus_Serious_Injry_Cnt', 'Nonincap_Injry_Cnt',
                 'Poss_Injry_Cnt', 'Non_Injry_Cnt', 'Unkn_Injry_Cnt', 'Tot_Injry_Cnt',
                 'Death_Cnt', 'Time_PM_peak', 'Time_Mid_day', 'Time_AM_peak',
                 'Time_Night/Early_Morning', 'Day_Friday', 'Day_Sunday', 'Day_Thursday',
                 'Day_Monday', 'Day_Wednesday', 'Day_Saturday', 'Day_Tuesday']

    df = df[keep_cols]
    return df


def load_map_crash2segment(df_crash, file_name):
    """ Load map of collisions to road segments.
    """
    c_map_crash2street = pd.read_csv(f'{DATA_PATH}/{file_name}')
    dict_map_crash2street = dict(zip(c_map_crash2street.Crash_ID.values, c_map_crash2street.STR_UNQ_ID.values))
    df_crash['STR_UNQ_ID'] = df_crash.Crash_ID.map(lambda x: dict_map_crash2street[x])
    return c_map_crash2street


def pickel_model(model_name, model, cols):
    """ Save CatBoost model and used columns.
    """
    print(f'Model: {model_name} Save to: {PICKEL_PATH}/{model_name}.pkl')
    file_name = f"{PICKEL_PATH}/{model_name}.pkl"
    pickle.dump(model, open(file_name, "wb"))

    file_name = f"{PICKEL_PATH}/{model_name}_Columns.pkl"
    pickle.dump(cols, open(file_name, "wb"))


def load_model(model_name):
    """ Load CatBoost model and used columns.
    """
    file_name = f"{PICKEL_PATH}/{model_name}.pkl"
    model = pickle.load(open(file_name, "rb"))

    file_name = f"{PICKEL_PATH}/{model_name}_Columns.pkl"
    cols = pickle.load(open(file_name, "rb"))
    return model, cols
