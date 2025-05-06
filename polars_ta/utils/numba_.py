"""
Demo for using numba to implement rolling functions.
本文件是使用numba实现rolling的函数，演示用
"""
from typing import List

import numpy as np
from numba import jit
from numpy import full
from numpy.lib.stride_tricks import sliding_window_view
from polars import Series, Expr, struct, DataFrame

"""
Series.to_numpy的操作在调用之前做，这样可控一些
batches_i1_o1这一类的函数输入不支持Series，只支持numpy。设计成在map_batches转换更可控
"""


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def isnan(x):
    # https://github.com/numba/numba/issues/2919#issuecomment-747377615
    if int(x) == -9223372036854775808:
        return True
    else:
        return False


@jit(nopython=True, nogil=True, cache=True)
def full_with_window_size(arr, fill_value, dtype=None, window_size: int = 1):
    """创建一个更大的数组，填充后一截数据"""
    out = full(arr.shape[0] + window_size - 1, fill_value, dtype=dtype)
    out[window_size - 1:] = arr
    return out


@jit(nopython=True, nogil=True, cache=True)
def sliding_window_with_min_periods(arr, window_size: int, min_periods: int):
    """为rolling准备的数据，当数据长度不足时，用nan填充"""
    windows = sliding_window_view(arr, window_size)
    valid_counts = np.sum(~np.isnan(windows), axis=1)
    # 修改这一行，使用布尔索引而不是np.where
    result = windows.copy()
    result[valid_counts < min_periods] = np.nan
    return result


@jit(nopython=True, nogil=True, cache=True)
def _roll_1(x1: np.ndarray, window: int, min_periods: int, func):
    """TODO 只是模板演示，无法编译通过"""
    out1 = full_with_window_size(x1, np.nan, dtype=np.float64, window_size=window)
    a1 = sliding_window_with_min_periods(out1, window, min_periods)
    out1[:] = np.nan
    for i, v1 in enumerate(a1):
        if np.isnan(v1).all():
            continue
        out1[i] = func(v1)
    return out1[:x1.shape[0]]


@jit(nopython=True, nogil=True, cache=True)
def _roll_2(x1, x2, window, min_periods, func):
    """TODO 只是模板演示，无法编译通过"""
    out1 = full_with_window_size(x1, np.nan, dtype=np.float64, window_size=window)
    out2 = full_with_window_size(x2, np.nan, dtype=np.float64, window_size=window)
    a1 = sliding_window_with_min_periods(out1, window, min_periods)
    a2 = sliding_window_with_min_periods(out2, window, min_periods)
    out1[:] = np.nan
    for i, (v1, v2) in enumerate(zip(a1, a2)):
        if np.isnan(v1).all():
            continue
        if np.isnan(v2).all():
            continue
        out1[i] = func(v1, v2)
    return out1[:x1.shape[0]]


@jit(nopython=True, nogil=True, cache=True)
def _roll_3(x1, x2, x3, window, min_periods, func):
    """TODO 只是模板演示，无法编译通过"""
    out1 = full_with_window_size(x1, np.nan, dtype=np.float64, window_size=window)
    out2 = full_with_window_size(x2, np.nan, dtype=np.float64, window_size=window)
    out3 = full_with_window_size(x3, np.nan, dtype=np.float64, window_size=window)
    a1 = sliding_window_with_min_periods(out1, window, min_periods)
    a2 = sliding_window_with_min_periods(out2, window, min_periods)
    a3 = sliding_window_with_min_periods(out3, window, min_periods)
    out1[:] = np.nan
    for i, (v1, v2, v3) in enumerate(zip(a1, a2, a3)):
        if np.isnan(v1).all():
            continue
        if np.isnan(v2).all():
            continue
        if np.isnan(v3).all():
            continue
        out1[i] = func(v1, v2, v3)
    return out1[:x1.shape[0]]


def struct_to_numpy(xx, n: int, dtype=None):
    if dtype is None:
        return [xx.struct[i].to_numpy() for i in range(n)]
    else:
        return [xx.struct[i].to_numpy().astype(dtype) for i in range(n)]


def batches_i1_o1(x1: np.ndarray, func, *args, dtype=None) -> Series:
    return Series(func(x1, *args), nan_to_null=True, dtype=dtype)


def batches_i2_o1(xx: List[np.ndarray], func, *args, dtype=None) -> Series:
    return Series(func(*xx, *args), nan_to_null=True, dtype=dtype)


def batches_i1_o2(x1: np.ndarray, func, *args, dtype=None) -> Series:
    return DataFrame(func(x1, *args), nan_to_null=True).to_struct()


def batches_i2_o2(xx: List[np.ndarray], func, *args, dtype=None) -> Series:
    return DataFrame(func(*xx, *args), nan_to_null=True).to_struct()


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
    return struct([a, b]).map_batches(lambda xx: batches_i2_o1(struct_to_numpy(xx, 2), nb_roll_cov, n))
