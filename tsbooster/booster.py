# %%
import pandas as pd
from tsbooster.tsibble import Tsibble
from typing import List, Callable, Optional
import sklearn
from tsbooster.features.date import add_week_date_features
import matplotlib.pyplot as plt
from functools import reduce

# %%
%reload_ext autoreload
%autoreload 2

# %%
path = "~/Documents/kaggle/walmart/"


# %%
data = pd.read_csv(path + "train.csv", parse_dates=True).assign(
    Date=lambda x: pd.to_datetime(x["Date"])
)
data.head()

# %%
sub = data[(data["Store"].isin([1, 2])) & (data["Dept"] == 1)]
# %%
sub
# %%
#tsib = Tsibble(data, index="Date", keys=["Store", "Dept"], freq=pd.offsets.Day(7))

tsib = Tsibble(sub, index="Date", keys=["Store", "Dept"], freq=pd.offsets.Day(7))

# %%
class LightBoostingTransformer:
    def __init__(self, index, keys, target, horizon):
        self.index = index
        self.keys = keys
        self.horizon = horizon
        self.target = target

    def fit_transform(self, data: Tsibble):
        df = data.data.copy()
        df = add_week_date_features(df, data.index)
        df = add_horizons(df, self.keys, self.index, self.target, self.horizon, data.freq)
        return df



# %%
def get_horizons(frequency, horizon):
    horizon_series = np.arange(1, horizon+1)
    forecast_offsets = pd.to_timedelta(horizon_series * frequency)
    return forecast_offsets
# %%
get_horizons(tsib.data[tsib.index], tsib.freq, 2)
# %%
import numpy as np
def add_horizons(data, keys, index, target, horizon, frequency):
    forecast_dates = pd.DataFrame(get_horizons(frequency, horizon), columns=["horizon"])
    data_w_horizon = data.merge(forecast_dates, how = "cross").assign(forecast_date = lambda x: x[index] + x["horizon"])
    target_df = data.loc[:, keys + [index] + [target]].copy()
    return (
        data_w_horizon
        .drop(target, axis=1)
        .rename(columns={index: "forecast_origin"})
        .merge(
            target_df,
            left_on=keys + ["forecast_date"],
            right_on=keys + [index]
        )
        .drop([index] + ["forecast_date"], axis=1)
        .rename(columns={"forecast_origin": index})
        .assign(horizon = lambda x: x["horizon"].dt.days)
    )
# %%
add_horizons(tsib.data, tsib.keys, tsib.index, "Weekly_Sales", 2, tsib.freq)

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
            LightBoostingTransformer(data.index, data.keys, target, horizon)
            if transformer is None
            else transformer
        )
        test_start = tsib.data[tsib.index].unique()[-horizon]
        self.cv = TimeseriesHoldout(data.index, test_start) if cv is None else cv # Update this to work with new horizon setup
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

        self.forecaster = self.modelselector.fit(self.X, self.y)
        return self.forecaster

    def predict(self, data: Tsibble):
        #test that new data index matches horizon using old data
        tranformed_data = self.transformer.fit_transform(data)
        X = transformed_data

        preds = self.forecaster.predict(X)
        return preds

    def split(self):
        if self.X is None and self.y is None:
            raise Exception("Cannot generate split before data has been transformed.")
        return self.cv.split(self.X, self.y)


# %%
booster = LightTSBooster(data=tsib, target="Weekly_Sales", horizon=4)
model = booster.fit()

# %%
np.sqrt(model.best_score_ * -1) / tsib.data["Weekly_Sales"].mean()

# %%
from tsbooster.adversarial_validation import AdversarialValidator

# %%
av = AdversarialValidator(cols_to_drop=["Date", "day_of_month", "week", "month"])
av.validate(
        booster.X.iloc[train_idx],
        booster.X.iloc[test_idx]
    )

# %%
av.feature_importances_()
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

