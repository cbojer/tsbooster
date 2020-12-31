# %%
from typing import List
import pandas as pd
from tsbooster.tsibble import Tsibble

# %%
def grouped_mean(data: pd.DataFrame, keys: List[str], target_col: str):
    return data.groupby(keys)[target_col].mean().to_frame(name=f"Avg_{target_col}")


def add_grouped_mean(data: pd.DataFrame, keys: List[str], target_col: str):
    gp_mean = grouped_mean(data, keys, target_col)
    data = data.copy()
    return data.merge(gp_mean, on=keys)