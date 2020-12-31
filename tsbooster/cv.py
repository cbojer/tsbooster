import pandas as pd
import numpy as np


class TimeseriesHoldout:
    def __init__(self, date_column: str, test_start: pd.Timestamp):
        self.date_column = date_column
        self.test_start = test_start

    def split(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray = None):
        train_indices = np.where(X[self.date_column] < self.test_start)[0]
        test_indices = np.where(X[self.date_column] >= self.test_start)[0]
        yield train_indices, test_indices
