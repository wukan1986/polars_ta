from functools import lru_cache

import numpy as np
import pandas as pd
import polars as pl
from numba import jit


@lru_cache
@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def get_window_bounds(
        num_values: int = 0,
        window_size: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    print(num_values, window_size)
    end = np.arange(1, num_values + 1, dtype=np.int64)
    start = end - window_size
    start = np.clip(start, 0, num_values)
    return start, end


def roll_rank(x: pl.Series, d: int, pct: bool = True, method: str = 'average', ascending: bool = True):
    start, end = get_window_bounds(len(x), d)
    """
    https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/window/aggregations.pyx#L1281
    
    def roll_rank(const float64_t[:] values, ndarray[int64_t] start,
              ndarray[int64_t] end, int64_t minp, bint percentile,
              str method, bint ascending) -> np.ndarray:
              
    O(N log(window)) implementation using skip list
    """
    ret = pd._libs.window.aggregations.roll_rank(x.to_numpy(), start, end, d, pct, method, ascending)
    return pl.Series(ret, nan_to_null=True)


def roll_kurt(x, d):
    start, end = get_window_bounds(len(x), d)
    """
    https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/window/aggregations.pyx#L803

    def roll_kurt(ndarray[float64_t] values, ndarray[int64_t] start,
              ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    """
    ret = pd._libs.window.aggregations.roll_kurt(x.to_numpy(), start, end, d)
    return pl.Series(ret, nan_to_null=True)
