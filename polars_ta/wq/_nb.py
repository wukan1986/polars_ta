import numpy as np
from numba import jit
from numpy.lib.stride_tricks import sliding_window_view


@jit(nopython=True, nogil=True, cache=True)
def nb_roll_argmax(x1, window):
    out = np.full(x1.shape, 0, dtype=int)
    a1 = sliding_window_view(x1, window)[:, ::-1]  # 注意倒序
    for i, v1 in enumerate(a1):
        out[i + window - 1] = np.argmax(v1)
    return out


@jit(nopython=True, nogil=True, cache=True)
def nb_roll_argmin(x1, window):
    out = np.full(x1.shape, 0, dtype=int)
    a1 = sliding_window_view(x1, window)[:, ::-1]  # 注意倒序
    for i, v1 in enumerate(a1):
        out[i + window - 1] = np.argmin(v1)
    return out


@jit(nopython=True, nogil=True, cache=True)
def nb_roll_prod(x1, window):
    out = np.full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    for i, v1 in enumerate(a1):
        out[i + window - 1] = np.prod(v1)
    return out
