"""
本文件是使用numba实现rolling的函数，演示用
"""
import numpy as np
from numba import jit
from numpy.lib.stride_tricks import sliding_window_view
from polars import Series, Expr, map_batches


def batches_1(x1: Series, windows: int, func1, *args, dtype=None) -> Series:
    return Series(func1(x1.to_numpy(), windows, *args), nan_to_null=True, dtype=dtype)


def batches_2(x12: Series, windows: int, func2, *args, dtype=None) -> Series:
    x1, x2 = x12
    return Series(func2(x1.to_numpy(), x2.to_numpy(), windows, *args), nan_to_null=True, dtype=dtype)


def batches_3(x123: Series, windows: int, func3, *args, dtype=None) -> Series:
    x1, x2, x3 = x123
    return Series(func3(x1.to_numpy(), x2.to_numpy(), x3.to_numpy(), windows, *args), nan_to_null=True, dtype=dtype)


@jit(nopython=True, nogil=True, cache=True)
def nb_roll_sum(x1, window):
    """演示代码，请直接用 pl.col('A').rolling_sum(10).alias('a1')"""
    out = np.full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    for i, v1 in enumerate(a1):
        out[i + window - 1] = np.sum(v1)
    return out


@jit(nopython=True, nogil=True, cache=True)
def nb_roll_cov(x1, x2, window):
    """演示代码，pl.rolling_cov(pl.col('A'), pl.col('B'), window_size=10).alias('a6')"""
    out = np.full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    a2 = sliding_window_view(x2, window)
    for i, (v1, v2) in enumerate(zip(a1, a2)):
        out[i + window - 1] = np.cov(v1, v2)[0, 1]
    return out


def roll_sum(x: Expr, n: int) -> Expr:
    return x.map_batches(lambda x1: batches_1(x1, n, nb_roll_sum))


def roll_cov(a: Expr, b: Expr, n: int) -> Expr:
    return map_batches([a, b], lambda x12: batches_2(x12, n, nb_roll_cov))
