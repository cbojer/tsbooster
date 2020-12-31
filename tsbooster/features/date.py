import pandas as pd
from tsbooster.tsibble import Tsibble


def add_date_features(data: pd.DataFrame, index: str):
    data["day_of_week"] = data[index].dt.dayofweek
    data["month"] = data[index].dt.month
    data["day_of_month"] = data[index].dt.day
    data["weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    return data


#%%
def add_week_date_features(data: pd.DataFrame, index: str):
    data["month"] = data[index].dt.month
    data["day_of_month"] = data[index].dt.day
    data["week"] = data[index].dt.week
    return data