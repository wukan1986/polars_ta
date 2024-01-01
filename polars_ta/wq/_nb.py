import numpy as np
from numba import jit
from numpy import argmax, argmin, prod, mean, std, full, vstack, corrcoef
from numpy.lib.stride_tricks import sliding_window_view


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def roll_argmax(x1, window, reverse):
    out = full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    if reverse:
        a1 = a1[:, ::-1]
    for i, v1 in enumerate(a1):
        out[i + window - 1] = argmax(v1)
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def roll_argmin(x1, window, reverse):
    out = full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    if reverse:
        a1 = a1[:, ::-1]
    for i, v1 in enumerate(a1):
        out[i + window - 1] = argmin(v1)
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def roll_prod(x1, window):
    out = full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    for i, v1 in enumerate(a1):
        out[i + window - 1] = prod(v1)
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _co_kurtosis(a1, a2):
    t1 = a1 - mean(a1)
    t2 = a2 - mean(a2)
    t3 = std(a1)
    t4 = std(a2)
    return mean(t1 * (t2 ** 3)) / (t3 * (t4 ** 3))


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def roll_co_kurtosis(x1, x2, window):
    out = full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    a2 = sliding_window_view(x2, window)
    for i, (v1, v2) in enumerate(zip(a1, a2)):
        out[i + window - 1] = _co_kurtosis(v1, v2)
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _co_skewness(a1, a2):
    t1 = a1 - mean(a1)
    t2 = a2 - mean(a2)
    t3 = std(a1)
    t4 = std(a2)
    return mean(t1 * (t2 ** 2)) / (t3 * (t4 ** 2))


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def roll_co_skewness(x1, x2, window):
    out = full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    a2 = sliding_window_view(x2, window)
    for i, (v1, v2) in enumerate(zip(a1, a2)):
        out[i + window - 1] = _co_skewness(v1, v2)
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _moment(a1, k):
    """中心矩阵"""
    return mean((a1 - mean(a1)) ** k)


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def roll_moment(x1, window, k):
    out = full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    for i, v1 in enumerate(a1):
        out[i + window - 1] = _moment(v1, k)
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _partial_corr(a1, a2, a3):
    """TODO 不知是否正确，需要检查"""
    c = corrcoef(vstack((a1, a2, a3)))
    rxy = c[0, 1]
    rxz = c[0, 2]
    ryz = c[1, 2]
    t1 = rxy - rxz * ryz
    t2 = (1 - rxz ** 2) ** 0.5
    t3 = (1 - ryz ** 2) ** 0.5
    return t1 / (t2 * t3)


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def roll_partial_corr(x1, x2, x3, window):
    out = full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    a2 = sliding_window_view(x2, window)
    a3 = sliding_window_view(x3, window)
    for i, (v1, v2, v3) in enumerate(zip(a1, a2, a3)):
        out[i + window - 1] = _partial_corr(v1, v2, v3)
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _triple_corr(a1, a2, a3):
    t1 = a1 - mean(a1)
    t2 = a2 - mean(a2)
    t3 = a3 - mean(a3)
    t4 = std(a1)
    t5 = std(a2)
    t6 = std(a3)
    return mean(t1 * t2 * t3) / (t4 * t5 * t6)


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def roll_triple_corr(x1, x2, x3, window):
    out = full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    a2 = sliding_window_view(x2, window)
    a3 = sliding_window_view(x3, window)
    for i, (v1, v2, v3) in enumerate(zip(a1, a2, a3)):
        out[i + window - 1] = _triple_corr(v1, v2, v3)
    return out
