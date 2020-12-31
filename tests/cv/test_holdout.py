#%%
from tsbooster.cv import TimeseriesHoldout
import pandas as pd
import numpy as np

#%%
def test_holdout_cv():
    data = {
        "time": np.arange(0, 30),
        "vals": np.arange(10, 40),
        "dates": pd.date_range(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-30"), freq="1 d"
        ).values,
    }
    df = pd.DataFrame(data)

    X = df.drop("vals", axis=1)
    y = df["vals"]
    test_start = pd.Timestamp("2020-01-16")

    cv = TimeseriesHoldout(date_column="dates", test_start=test_start)
    splits = cv.split(X, y)
    train_idx, test_idx = next(splits)
    exp_train_idx = np.arange(0, 15)
    exp_test_idx = np.arange(15, 30)

    # Test length is as expected (to prevent wierd hard-to-interpret boolean error in case they are not)
    assert len(train_idx) == len(exp_train_idx)
    assert len(test_idx) == len(exp_test_idx)

    # Assert the indexes are as expected
    assert (train_idx == exp_train_idx).all()
    assert (test_idx == exp_test_idx).all()

    # Assert the Dates are as expected if we split using the indices
    assert (X.iloc[train_idx]["dates"] < test_start).all()
    assert (X.iloc[test_idx]["dates"] >= test_start).all()

    # Assert the y-values are as expected if we split using the indices
    assert (y[train_idx] == np.arange(10, 25)).all()
    assert (y[test_idx] == np.arange(25, 40)).all()
