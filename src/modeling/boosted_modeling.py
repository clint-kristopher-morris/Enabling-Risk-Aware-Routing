from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.model_selection import train_test_split


def split_data(df, label, split_size):
    """ Test train split data.
    """
    x = df.drop([label], 1)
    y = df[label]
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=split_size)  # split data
    return x_train, x_valid, y_train, y_valid


def train_catboost(params, x_train, x_valid, y_train, y_valid):
    """ Conducts CatBoost modeling.
    """
    train_data = Pool(data=x_train,
                      label=y_train)
    valid_data = Pool(data=x_valid,
                      label=y_valid)
    model = CatBoostClassifier(**params)
    model.fit(train_data,
              eval_set=valid_data,
              use_best_model=True,
              plot=True)
    return model
