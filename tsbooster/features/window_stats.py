from typing import List
import pandas as pd
from tsbooster.tsibble import Tsibble
import numpy as np


def add_moving_averages(tsib: Tsibble, windows: List[int], target: str):
    data = tsib.data.copy()
    for w in windows:
        ma = (
            data.set_index(tsib.index)
            .groupby(tsib.keys)[target]
            .rolling(w)
            .mean()
            .to_frame(f"MA_{w}_{target}")
        )
        data = data.merge(ma, on=tsib.keys + [tsib.index])
    return data


def add_exp_moving_averages(tsib: Tsibble, windows: List[int], target: str):
    data = tsib.data.copy()
    for w in windows:
        data[f"EWMA_{w}_{target}"] = (
            data.set_index(tsib.index)
            .groupby(tsib.keys)[target]
            .transform(
                lambda x: x.ewm(span=w, min_periods=int(np.ceil(w ** 0.8))).mean()
            )
            .values
        )
    return data


def quantile90(x):
    return x.quantile(0.9)


def quantile10(x):
    return x.quantile(0.1)


# %%
def add_moving_stats(
    tsib: Tsibble,
    windows: List[int],
    target: str,
    stats: List[str],
    group_keys: List[str] = [],
):
    data = tsib.data
    for w in windows:
        m_stats = (
            tsib.data.set_index(tsib.index)
            .groupby(tsib.keys + group_keys)[target]
            .rolling(w)
            .agg(stats)
        )
        m_stats.columns = [f"{s}_{w}_{target}" for s in stats]
        data = data.merge(m_stats, on=tsib.keys + [tsib.index] + group_keys)
    return data
