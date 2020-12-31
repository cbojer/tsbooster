#%%
from typing import List
import pandas as pd

# %%
class Tsibble:
    def __init__(self, data: pd.DataFrame, index: str, keys: List[str], freq: str):
        self.data = data
        self.validate_index(index)
        self.validate_keys(keys)
        self.keys = keys  # TODO: should be made optional
        self.index = index
        self.data = data.sort_values(index)
        self.freq = freq

    def validate_index(self, index: str):
        if not index in self.data.columns:
            raise Exception(
                "The given Index name must be present as a column in the dataset"
            )

    def validate_keys(self, keys: List[str]):
        for key in keys:
            if not key in self.data.columns:
                raise Exception(
                    f"The given key ({key}) is not present as a column in the dataset"
                )

    def is_regular(self) -> bool:
        full_counts = self.data.groupby(self.keys).apply(
            lambda x: len(self._get_full_date_range(x[self.index]))
        )
        act_counts = self.data.groupby(self.keys)[self.index].nunique()
        counts = full_counts.to_frame("full").join(act_counts.to_frame("act"))
        irregular = counts.query("full != act")
        return len(counts.index) == 0

    def _get_full_date_range(self, dates):
        return pd.date_range(dates.min(), dates.max(), freq=self.freq)
