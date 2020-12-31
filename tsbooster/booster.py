# %%
import pandas as pd
from tsbooster.tsibble import Tsibble
from typing import List, Callable, Optional
import sklearn
from tsbooster.features.date import add_week_date_features


# %%
path = "~/Documents/kaggle/walmart/"


# %%
data = pd.read_csv(path + "train.csv", parse_dates=True).assign(
    Date=lambda x: pd.to_datetime(x["Date"])
)
data.head()

# %%
tsib = Tsibble(data, index="Date", keys=["Store", "Dept"], freq=pd.offsets.Day(7))


# %%
from functools import reduce


# %%
class LightBoostingTransformer:
    def __init__(self, index, keys, horizon):
        self.index = index
        self.keys = keys
        self.horizon = horizon

    def fit_transform(self, data: Tsibble):
        df = data.data.copy()
        df = add_week_date_features(df, data.index)
        return df


#%%
transformer = LightBoostingTransformer(index="Date", keys=["Store", "Dept"], horizon=4)

#%%
df = transformer.fit_transform(tsib)
df


# %%
from typing import Iterable, Tuple
from tsbooster.cv import TimeseriesHoldout
from sklearn.model_selection import GridSearchCV

#%%
class LightTSBooster:
    def __init__(
        self,
        data: Tsibble,
        target: str,
        horizon: int,
        transformer=None,
        cv=None,
        modelselector=None,
    ):
        self.data = data
        self.target = target
        self.horizon = horizon
        self.transformer = (
            LightBoostingTransformer(data.index, data.keys, horizon)
            if transformer is None
            else transformer
        )
        test_start = tsib.data[tsib.index].unique()[-horizon]
        self.cv = TimeseriesHoldout(data.index, test_start) if cv is None else cv
        self.modelselector = ()

    def fit(self):
        transformed_data = self.transformer.fit_transform(self.data)
        y = transformed_data[self.target]
        X = transformed_data.drop([self.target], axis=1)

        for train_idx, test_idx in self.cv.split(X, y):
            print(len(train_idx), len(test_idx))
        return X, y


# %%
# %%
booster = LightTSBooster(data=tsib, target="Weekly_Sales", horizon=4)


# %%
booster.fit()

#%%
import lightgbm as lgb


lgb.LGBMRegressor()
