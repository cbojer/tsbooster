#%%
import numpy as np
import pandas as pd
from tsbooster.cv import TimeseriesHoldout

#%%
data = {
    "time": np.arange(0, 30),
    "vals": np.random.normal(size=30),
    "dates": pd.date_range(
        pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-30"), freq="1 d"
    ).values,
}
df = pd.DataFrame(data)

X = df.drop("vals", axis=1)
y = df["vals"]

cv = TimeseriesHoldout(date_column="dates", test_start=pd.Timestamp("2020-01-16"))
splits = cv.split(X, y)
train_idx, test_idx = next(splits)

# %%
train_idx == np.arange(0, 15)
# %%
test_idx == np.arange(15, 30)
# %%
len(test_idx)

#%%
np.arange(15, 31)
#%%

assert (train_idx == np.arange(0, 15)).all()
assert (test_idx == np.arange(15, 31)).all()