"""
Demo for using numba to implement rolling functions.
本文件是使用numba实现rolling的函数，演示用
"""
from typing import List

import numpy as np
from numba import jit
from numpy.lib.stride_tricks import sliding_window_view
from polars import Series, Expr, struct, DataFrame

"""
Series.to_numpy的操作在调用之前做，这样可控一些
batches_i1_o1这一类的函数输入不支持Series，只支持numpy。设计成在map_batches转换更可控
"""


def batches_i1_o1(x1: np.ndarray, func, *args, dtype=None) -> Series:
    return Series(func(x1, *args), nan_to_null=True, dtype=dtype)


def batches_i2_o1(xx: List[np.ndarray], func, *args, dtype=None) -> Series:
    return Series(func(*xx, *args), nan_to_null=True, dtype=dtype)


def batches_i1_o2(x1: np.ndarray, func, *args, dtype=None) -> Series:
    out = func(x1, *args)
    ss = [Series(x, nan_to_null=True, dtype=dtype) for x in out]
    return DataFrame(ss).to_struct()


def batches_i2_o2(xx: List[np.ndarray], func, *args, dtype=None) -> Series:
    out = func(*xx, *args)
    ss = [Series(x, nan_to_null=True, dtype=dtype) for x in out]
    return DataFrame(ss).to_struct()


def batches_i2_o2_v2(xx: List[np.ndarray], func, *args, dtype=None) -> Series:
    """此写法也能用，速度差异不大"""
    out = func(*xx, *args)
    arr = np.empty((xx[0].shape[0], len(out)), dtype=dtype)
    for i, x in enumerate(out):
        arr[:, i] = x
    return Series(arr, nan_to_null=True).arr.to_struct()


@jit(nopython=True, nogil=True, cache=True)
def nb_roll_sum(x1, window):
    """Demo code. Use `pl.col('A').rolling_sum(10).alias('a1')` instead.
    演示代码，请直接用 pl.col('A').rolling_sum(10).alias('a1')"""
    out = np.full(x1.shape, np.nan, dtype=np.float64)
    if len(x1) < window:
        return out
    a1 = sliding_window_view(x1, window)
    for i, v1 in enumerate(a1):
        out[i + window - 1] = np.sum(v1)
    return out


@jit(nopython=True, nogil=True, cache=True)
def nb_roll_cov(x1, x2, window):
    """Demo code. Use `pl.rolling_cov(pl.col('A'), pl.col('B'), window_size=10).alias('a6')` instead.
    演示代码，pl.rolling_cov(pl.col('A'), pl.col('B'), window_size=10).alias('a6')"""
    out = np.full(x1.shape, np.nan, dtype=np.float64)
    if len(x1) < window:
        return out
    a1 = sliding_window_view(x1, window)
    a2 = sliding_window_view(x2, window)
    for i, (v1, v2) in enumerate(zip(a1, a2)):
        out[i + window - 1] = np.cov(v1, v2)[0, 1]
    return out


def roll_sum(x: Expr, n: int) -> Expr:
    return x.map_batches(lambda x1: batches_i1_o1(x1.to_numpy(), nb_roll_sum, n))


def roll_cov(a: Expr, b: Expr, n: int) -> Expr:
    return struct([a, b]).map_batches(lambda xx: batches_i2_o1([xx.struct[i].to_numpy() for i in range(2)], nb_roll_cov, n))
