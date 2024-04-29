"""
Demo for using numba to implement rolling functions.
本文件是使用numba实现rolling的函数，演示用
"""
from typing import List

import numpy as np
from numba import jit
from numpy.lib.stride_tricks import sliding_window_view
from polars import Series, Expr, map_batches

"""
Series.to_numpy的操作在调用之前做，这样可控一些
batches_i1_o1这一类的函数输入不支持Series，只支持numpy。设计成在map_batches转换更可控
"""


def batches_i1_o1(x1: np.ndarray, func, *args, dtype=None) -> Series:
    return Series(func(x1, *args), nan_to_null=True, dtype=dtype)


def batches_i2_o1(xx: List[np.ndarray], func, *args, dtype=None) -> Series:
    return Series(func(*xx, *args), nan_to_null=True, dtype=dtype)


def batches_i1_o2(x1: np.ndarray, func, *args, dtype=None, ret_idx: int = 0) -> Series:
    return Series(func(x1, *args)[ret_idx], nan_to_null=True, dtype=dtype)


def batches_i2_o2(xx: List[np.ndarray], func, *args, dtype=None, ret_idx: int = 0) -> Series:
    return Series(func(*xx, *args)[ret_idx], nan_to_null=True, dtype=dtype)


@jit(nopython=True, nogil=True, cache=True)
def nb_roll_sum(x1, window):
    """Demo code. Use `pl.col('A').rolling_sum(10).alias('a1')` instead.
    演示代码，请直接用 pl.col('A').rolling_sum(10).alias('a1')"""
    out = np.full(x1.shape, np.nan, dtype=float)
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
    out = np.full(x1.shape, np.nan, dtype=float)
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
    return map_batches([a, b], lambda xx: batches_i2_o1([x1.to_numpy() for x1 in xx], nb_roll_cov, n))


@jit(nopython=True, nogil=True, cache=True)
def nb_split_o1(x1, x2, window=10, n=2):
    out = np.full((x1.shape[0], 2), np.nan, dtype=float)
    if len(x1) < window:
        return out[:, 1] - out[:, 0]
    a1 = sliding_window_view(x1, window)
    a2 = sliding_window_view(x2, window)
    for i, (v1, v2) in enumerate(zip(a1, a2)):
        # 排序两次，解决nan的问题
        b1 = np.argsort(v2)[:n]
        b2 = np.argsort(-v2)[:n]
        out[i + window - 1, 0] = np.sum(v1[b1])
        out[i + window - 1, 1] = np.sum(v1[b2])
    return out[:, 1] - out[:, 0]


@jit(nopython=True, nogil=True, cache=True)
def nb_split_o2(x1, x2, window=10, n=2):
    out = np.full((x1.shape[0], 2), np.nan, dtype=float)
    if len(x1) < window:
        return out
    a1 = sliding_window_view(x1, window)
    a2 = sliding_window_view(x2, window)
    for i, (v1, v2) in enumerate(zip(a1, a2)):
        # 排序两次，解决nan的问题
        b1 = np.argsort(v2)[:n]
        b2 = np.argsort(-v2)[:n]
        out[i + window - 1, 0] = np.sum(v1[b1])
        out[i + window - 1, 1] = np.sum(v1[b2])
    return out


def roll_split_i2_o1(a: Expr, b: Expr, d: int, n: int) -> Expr:
    """切割论的示例。只输出一个结果"""
    return map_batches(
        [a, b],
        lambda xx: batches_i2_o1([x1.to_numpy() for x1 in xx], nb_split_o1, d, n)
    )


def roll_split_i2_o2(a: Expr, b: Expr, d: int, n: int) -> Expr:
    """切割论的示例。两输出，由用户自己进行后期处理"""
    # !!!输出为struct时的特殊写法，使用其它写法都会导致之后想struct.field失败
    return map_batches(
        [a, b],
        lambda xx: batches_i2_o1([x1.to_numpy() for x1 in xx], nb_split_o2, d, n).list.to_struct()
    ).struct.rename_fields(['split_a', 'split_b'])
