# %%
import pandas as pd
from tsbooster.tsibble import Tsibble
from typing import List, Callable, Optional
import sklearn
from tsbooster.features.date import add_week_date_features
import matplotlib.pyplot as plt
from functools import reduce



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
%reload_ext autoreload
%autoreload 2
# %%
from typing import Iterable, Tuple
from tsbooster.cv import TimeseriesHoldout
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#%%
class LightTSBooster:
    def __init__(
        self,
        data: Tsibble,
        target: str,
        horizon: int,
        transformer=None,
        cv=None,
        model=None,
        modelselector=None,
        param_grid=None,
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
        self.model = lgb.LGBMRegressor() if model is None else model
        self.pipeline = (
            Pipeline(steps=[
                ("drop_date", ColumnTransformer([("drop", "drop", [data.index])], remainder="passthrough")),
                ("model", self.model)
            ])
        )

        self.params = (
            {"model__subsample": [0.5, 0.75, 1.0], "model__colsample_bytree": [0.5, 0.75, 1.0], "model__num_leaves": [2**5-1, 2**6-1, 2**7-1], "model__learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5]}
            if param_grid is None
            else param_grid
        )
        self.modelselector = (
            RandomizedSearchCV(self.pipeline, self.params, cv=self.cv, scoring="neg_mean_squared_error")
            if modelselector is None
            else modelselector
        )
        self.X = None
        self.y = None

    def fit(self):
        transformed_data = self.transformer.fit_transform(self.data)
        self.y = transformed_data[self.target]
        self.X = transformed_data.drop([self.target], axis=1)

        models = self.modelselector.fit(self.X, self.y)
        return models


# %%
booster = LightTSBooster(data=tsib, target="Weekly_Sales", horizon=4)
model = booster.fit()

# %%
lgbm = model.best_estimator_.named_steps["model"]
lgb.plot_importance(lgbm)

# %%
model.cv_results_
# %%
model.best_params_, model.best_score_
#%%
train_idx, test_idx = next(booster.cv.split(booster.X, booster.y))

#%%
preds = model.predict(booster.X.iloc[train_idx])
# %%
(booster.y.iloc[train_idx] - preds).abs().mean()

# %%
from math import sqrt
preds_y = model.predict(booster.X.iloc[test_idx])
((booster.y.iloc[test_idx] - preds_y)**2).mean()
