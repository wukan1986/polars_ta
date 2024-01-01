import numpy as np
from numba import jit
from numpy import mean, abs, full, argmax
from numpy.lib.stride_tricks import sliding_window_view


@jit(nopython=True, nogil=True, cache=True)
def roll_avedev(x1, window):
    out = full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    for i, v1 in enumerate(a1):
        out[i + window - 1] = mean(abs(v1 - mean(v1)))
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def roll_bars_since_n(x1, window):
    """BARSSINCEN(X,N):N周期内第一次X不为0到现在的天数

    TODO 如果一个周期内，都不满足，值取多少？0表当前值满足条件, window-1表示的是区间第0位置的值
    TODO 用window来表示都不满足
    """
    out = full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    for i, v1 in enumerate(a1):
        p = argmax(v1)
        out[i + window - 1] = window - 1 - p if p or v1[0] else window
    return out
