import numpy as np
from numba import jit
from numpy import mean, abs
from numpy.lib.stride_tricks import sliding_window_view


@jit(nopython=True, nogil=True, cache=True)
def nb_roll_avedev(x1, window):
    out = np.full(x1.shape, np.nan, dtype=float)
    a1 = sliding_window_view(x1, window)
    for i, v1 in enumerate(a1):
        out[i + window - 1] = mean(abs(v1 - mean(v1)))
    return out
