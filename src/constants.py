import numpy as np
import pickle

DATA_PATH ='data'
PICKEL_PATH = f'{DATA_PATH}/models'
ROAD_DATA_PATH = f'{DATA_PATH}/pdStreets.csv'
ROAD2WEATHER2DATE_PATH = f'{DATA_PATH}/NonCrashRoadsDayMonth.csv'
PATH2CRASH_DIR = f'{DATA_PATH}/txdot_crash_data'
PATH2WEATHER_OBJ_DIR = 'non_crash_data/weather'
PATH2OUTPUT_CSV_NC_DIR = 'obj/NonCrashWeather/csv'
PUBISHED_TRAFFIC_DATA_MONTHLY_FILE_NAME = 'monthly_traffic_aves.csv'
PUBISHED_TRAFFIC_DATA_HOURLY_FILE_NAME = 'DailyHourly.csv'
CLEANED_NC_FILE_PATH = f'{DATA_PATH}/processed/joinedNC_dataClean.csv'
CLEANED_C_FILE_PATH = f'{DATA_PATH}/processed/joinedCrash_dataClean.csv'
WEATHER_LOCATIONS_PATH = f'{DATA_PATH}/weather_station_locations.csv'
CRASH_DATA_FILE_NAME = 'crash_and_weather_data_n1473811.csv'

# plotting
FIGSIZE = (15, 3)
T_GRAY = (0.5, 0.5, 0.5)
T_RED = (0.8, 0, 0)
MAP_PLOT_SIZE = (20, 10)
WEB_MERCATOR_EPSG = 3857
LAT_LON_EPSG = 4326
TARGET_STATE = 'Texas'
WEATHER_STATION_ICON_PATH = 'static/icons/weather_icon_pin.svg'

# constants
COST_ARR = np.array([98400, 28500, 23400, 4600, 12500, 1704000])
DAYS = ['Day_Monday','Day_Tuesday','Day_Wednesday','Day_Thursday','Day_Friday','Day_Sunday','Day_Saturday']
NUM = [x for x in range(len(DAYS))]
NUM2DAY = dict(zip( NUM, DAYS ))
SEVERITY_DIS = pickle.load(open("data/models/severityDis.pkl", "rb"))
DISTRO_CRASH = pickle.load(open("data/models/distroCrash.pkl", "rb"))
DISTRO_SEVER = SEVERITY_DIS.values/sum(SEVERITY_DIS)

# demo
FREDERICKSBURG_ID = 15150
DEMO_LOCATION = 'Fredericksburg, TX, USA'
MAP_DATA_PATH = 'data/map_data/TxDOT_Roadway_Inventory_2019/TxDOT_Roadway_Inventory_2019.shp'

# weather data
API_KEY = ''
COLLECTION_SAVE_THRESH = 10000  # How many weather data points to collect before saving

USELESS_ROAD_COLS = ['Shape__Len', 'geometry', 'Unnamed: 0', 'UAN_HPMS', 'UAN', 'MPA', 'STE_NAM', 'TO_DISP', 'TO_NUM',
                     'RIA_RTE_ID', 'FRM_DFO', 'TO_DFO', 'HPMSID', 'RTE_GRID', 'GID', 'ACCEL_DECE', 'LEN_SEC',
                     'ADT_HIST_Y', 'HY_2', 'HY_3', 'HY_4', 'HY_5', 'HY_6', 'HY_7', 'HY_8', 'HY_9', 'TRUCK_HY_2',
                     'TRUCK_HY_3', 'TRUCK_HY_4', 'TRUCK_HY_5', 'TRUCK_HY_6', 'TRUCK_HY_7', 'TRUCK_HY_8', 'TRUCK_HY_9',
                     'TRF_STA_ID', 'HOV_TYP', 'DOTT', 'HWY_STAT_D', 'TOP100ID', 'MAINT_DIS', 'MSA_CNTY', 'TO_MKR_DAT',
                     'FRM_MKR_DA', 'TO_NBR', 'FRM_DISP', 'FRM_NUM', 'FRM_NBR', 'OBJECTID', 'REC', 'BMP', 'EMP',
                     'RDBD_ID', 'MNT_SEC', 'C_SEC', 'SEC_NHS', 'RI_MPT_DAT', 'HWY', 'SEC_NHS_AP', 'SEC_STR',
                     'SEC_TRUNK', 'LOAD_AXLE', 'LOAD_GROSS', 'LOAD_TAND', 'SEC_ADP', 'SEC_BIC', 'SEC_PARK', 'SEC_TTT',
                     'SEC_STM', 'SEC_NFH', 'SEC_TRUNK', 'SEC_TRK', 'SEC_HAZ', 'SEC_EVAC', 'SEC_FED_AI', 'HWY_DES1',
                     'ATH_100', 'ATH_PCT', 'ADT_YEAR', 'BRDG_STRUC', 'CON', 'PBLC_LAND', 'ADMIN', 'RDWAY_MAIN', 'SEC',
                     'HNUM', 'HSUF', 'ACES_CTRL', 'FLEX_ESAL', 'RIGID_ESAL', 'PHY_RDBD', 'SURF_TREAT', 'SURF_TRE_1',
                     'SURF_TRE_2', 'BASE_TP', 'DI', 'CO', 'CITY', 'SPD_MAX', 'SPD_MIN', 'F_SYSTEM', 'SRF_TYPE',
                     'S_TYPE_I', 'S_TYPE_O', 'MSA_CLS', 'ADT_CUR']

INJRY_COLS = ['Non_Injry_Cnt', 'Poss_Injry_Cnt', 'Unkn_Injry_Cnt', 'Nonincap_Injry_Cnt', 'Sus_Serious_Injry_Cnt',
              'Death_Cnt']

# beta model
MODEL_BETA_PARAMS = {'learning_rate': 0.01, 'eval_metric': 'AUC', 'max_depth': 10, 'early_stopping_rounds': 100,
                     'verbose': 200, 'random_seed': 42}

BST_COLS_BETA_MODEL = ['wspd', 'Time_PM_peak', 'Time_AM_peak', 'AADT_TRUCK', 'RU_1', 'HSYS_CR',
    'Time_Night/Early_Morning', 'LN_MILES', 'PCT_PK_CUT', 'precip_hrly', 'Day_Sunday', 'INCRS_FCTR', 'RU_2', 'D_FAC',
    'vis', 'HSYS_IH', 'HSYS_TL', 'MED_TYPE_5.0', 'HSYS_RM', 'Day_Saturday', 'HSYS_FM', 'HSYS_US', 'HSYS_SL',
                       # 'HSYS_BS','HSYS_UA'
    'MED_TYPE_3.0']

FCLASS2STREET_MAP = {'R1': 'R1', 'R2': 'R1', 'R3': 'R4', 'R4': 'R4', 'R5': 'RO', 'R6': 'RO', 'R7': 'RO',

                     'U1': 'U1', 'U2': 'U1', 'U3': 'U4', 'U4': 'U4', 'U5': 'UO', 'U6': 'UO', 'U7': 'UO'}

ROAD_CLASS_COLS = ['U1', 'R1', 'U4', 'R4', 'UO', 'RO']

# model alpha
BST_COLS_ALPHA_MODEL = ['LN_MILES', 'wspd', 'Time_PM_peak', 'Time_AM_peak', 'vis', 'AADT_TRUCK', 'precip_hrly', 'RU_1',
                        'MED_WID', 'MED_TYPE_0.0', 'Time_Night/Early_Morning', 'NUM_LANES', 'RU_4', 'Time_Mid_day',
                        'Day_Sunday', 'MED_TYPE_3.0', 'ROW_MIN', 'S_WID_O', 'Day_Saturday', 'D_FAC']

MODEL_ALPHA_PARAS = {'learning_rate': 0.01, 'eval_metric': 'AUC', 'max_depth': 9, 'early_stopping_rounds': 100,
    'verbose': 200, 'random_seed': 42}


