from functools import lru_cache
from typing import Tuple

import numpy as np
from numba import jit
from pandas._libs.window.aggregations import roll_kurt as _roll_kurt
from pandas._libs.window.aggregations import roll_rank as _roll_rank
from polars import Series

"""
When converting float32 to float64 before computing. Either use
x.cast(Float64).to_numpy()
or
x.to_numpy().astype(float)

The second one is faster

在计算前需要将float32转成float64，有以下两种方法
x.cast(Float64).to_numpy()
x.to_numpy().astype(float)

第二种方法更快
"""


@lru_cache
@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def get_window_bounds(
        num_values: int = 0,
        window_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    end = np.arange(1, num_values + 1, dtype=np.int64)
    start = end - window_size
    start = np.clip(start, 0, num_values)
    return start, end


def roll_rank(x: Series, d: int, pct: bool = True, method: str = 'average', ascending: bool = True):
    start, end = get_window_bounds(len(x), d)
    """
    https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/window/aggregations.pyx#L1281

    def roll_rank(const float64_t[:] values, ndarray[int64_t] start,
              ndarray[int64_t] end, int64_t minp, bint percentile,
              str method, bint ascending) -> np.ndarray:

    O(N log(window)) implementation using skip list
    """
    ret = _roll_rank(x.to_numpy().astype(float), start, end, d, pct, method, ascending)
    return Series(ret, nan_to_null=True)


def roll_kurt(x, d):
    start, end = get_window_bounds(len(x), d)
    """
    https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/window/aggregations.pyx#L803

    def roll_kurt(ndarray[float64_t] values, ndarray[int64_t] start,
              ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    """
    ret = _roll_kurt(x.to_numpy().astype(float), start, end, d)
    return Series(ret, nan_to_null=True)
