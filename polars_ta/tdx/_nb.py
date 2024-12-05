import numpy as np
from numba import jit
from numpy import mean, abs, full, argmax
from numpy.lib.stride_tricks import sliding_window_view


@jit(nopython=True, nogil=True, cache=True)
def roll_avedev(x1, window):
    out = full(x1.shape, np.nan, dtype=np.float64)
    if len(x1) < window:
        return out
    a1 = sliding_window_view(x1, window)
    for i, v1 in enumerate(a1):
        out[i + window - 1] = mean(abs(v1 - mean(v1)))
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def roll_bars_since_n(x1, window):
    """BARSSINCEN(X,N): the distance of the first observation that `X != 0` in `N` periods
    BARSSINCEN(X,N):N周期内第一次X不为0到现在的天数

    TODO what if all values are 0?

    TODO 如果一个周期内，都不满足，值取多少？0表当前值满足条件, window-1表示的是区间第0位置的值
    TODO 用window来表示都不满足
    """
    out = full(x1.shape, np.nan, dtype=np.float64)
    if len(x1) < window:
        return out
    a1 = sliding_window_view(x1, window)
    for i, v1 in enumerate(a1):
        p = argmax(v1)
        out[i + window - 1] = window - 1 - p if p or v1[0] else window
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _up_stat(a, d: int = 3):
    """T天N板，最稀疏为5天2板

    最近几天涨停但当天没涨停也会有记录，如6天2板，所以要与涨停一起使用
    """
    out1 = full(a.shape, 0, dtype=np.int64)
    out2 = full(a.shape, 0, dtype=np.int64)
    out3 = full(a.shape, 0, dtype=np.int64)
    t = 0  # T天
    n = 0  # N板
    k = 0  # 连续False个数
    f = True  # 前面的False不处理
    for i in range(0, a.shape[0]):
        if a[i]:
            k = 0
            t += 1
            n += 1
            f = False
            out1[i] = t
            out2[i] = n
            out3[i] = k
        else:
            if f:
                continue
            k += 1
            t += 1
            if k > d:
                # 超过指定天数才会重置
                t = 0
                n = 0
            out1[i] = t
            out2[i] = n
            out3[i] = k
    return out1, out2, out3
