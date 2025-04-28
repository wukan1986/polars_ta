import numpy as np
from numba import jit, float64, boolean
from numpy import argmax, argmin, full, vstack, corrcoef, nanprod, nanmean, nanstd
from numpy.lib.stride_tricks import sliding_window_view

from polars_ta.utils.numba_ import isnan, full_with_window_size, sliding_window_with_min_periods


@jit(nopython=True, nogil=True, cache=True)
def roll_argmax(x1, window, min_periods, reverse):
    out1 = full_with_window_size(x1, np.nan, dtype=np.float64, window_size=window)
    a1 = sliding_window_with_min_periods(out1, window, min_periods)
    out1[:] = np.nan
    if reverse:
        a1 = a1[:, ::-1]
    for i, v1 in enumerate(a1):
        if np.isnan(v1).all():
            continue
        out1[i] = argmax(v1)
    return out1[:x1.shape[0]]


@jit(nopython=True, nogil=True, cache=True)
def roll_argmin(x1, window, min_periods, reverse):
    out1 = full_with_window_size(x1, np.nan, dtype=np.float64, window_size=window)
    a1 = sliding_window_with_min_periods(out1, window, min_periods)
    out1[:] = np.nan
    if reverse:
        a1 = a1[:, ::-1]
    for i, v1 in enumerate(a1):
        if np.isnan(v1).all():
            continue
        out1[i] = argmin(v1)
    return out1[:x1.shape[0]]


@jit(nopython=True, nogil=True, cache=True)
def roll_prod(x1, window, min_periods):
    out1 = full_with_window_size(x1, np.nan, dtype=np.float64, window_size=window)
    a1 = sliding_window_with_min_periods(out1, window, min_periods)
    out1[:] = np.nan
    for i, v1 in enumerate(a1):
        if np.isnan(v1).all():
            continue
        out1[i] = nanprod(v1)
    return out1[:x1.shape[0]]


@jit(nopython=True, nogil=True, cache=True)
def _co_kurtosis(a1, a2):
    t1 = a1 - nanmean(a1)
    t2 = a2 - nanmean(a2)
    t3 = nanstd(a1)
    t4 = nanstd(a2)
    return nanmean(t1 * (t2 ** 3)) / (t3 * (t4 ** 3))


@jit(nopython=True, nogil=True, cache=True)
def roll_co_kurtosis(x1, x2, window, min_periods):
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
        out1[i] = _co_kurtosis(v1, v2)
    return out1[:x1.shape[0]]


@jit(nopython=True, nogil=True, cache=True)
def _co_skewness(a1, a2):
    t1 = a1 - nanmean(a1)
    t2 = a2 - nanmean(a2)
    t3 = nanstd(a1)
    t4 = nanstd(a2)
    return nanmean(t1 * (t2 ** 2)) / (t3 * (t4 ** 2))


@jit(nopython=True, nogil=True, cache=True)
def roll_co_skewness(x1, x2, window, min_periods):
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
        out1[i] = _co_skewness(v1, v2)
    return out1[:x1.shape[0]]


@jit(nopython=True, nogil=True, cache=True)
def _moment(a1, k):
    """centeral moment
    中心矩阵"""
    return nanmean((a1 - nanmean(a1)) ** k)


@jit(nopython=True, nogil=True, cache=True)
def roll_moment(x1, window, min_periods, k):
    out1 = full_with_window_size(x1, np.nan, dtype=np.float64, window_size=window)
    a1 = sliding_window_with_min_periods(out1, window, min_periods)
    out1[:] = np.nan
    for i, v1 in enumerate(a1):
        if np.isnan(v1).all():
            continue
        out1[i] = _moment(v1, k)
    return out1[:x1.shape[0]]


@jit(nopython=True, nogil=True, cache=True)
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


@jit(nopython=True, nogil=True, cache=True)
def roll_partial_corr(x1, x2, x3, window, min_periods):
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
        out1[i] = _partial_corr(v1, v2, v3)
    return out1[:x1.shape[0]]


@jit(nopython=True, nogil=True, cache=True)
def _triple_corr(a1, a2, a3):
    t1 = a1 - nanmean(a1)
    t2 = a2 - nanmean(a2)
    t3 = a3 - nanmean(a3)
    t4 = nanstd(a1)
    t5 = nanstd(a2)
    t6 = nanstd(a3)
    return nanmean(t1 * t2 * t3) / (t4 * t5 * t6)


@jit(nopython=True, nogil=True, cache=True)
def roll_triple_corr(x1, x2, x3, window, min_periods):
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
        out1[i] = _triple_corr(v1, v2, v3)
    return out1[:x1.shape[0]]


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _cum_prod_by(r, by):
    out = by.copy()
    for i in range(1, r.shape[0]):
        if isnan(out[i]):
            out[i] = r[i] * out[i - 1]
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _cum_sum_by(r, by):
    out = by.copy()
    for i in range(1, r.shape[0]):
        if isnan(out[i]):
            out[i] = r[i] + out[i - 1]
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _cum_sum_reset(a):
    last = 0
    out = full(a.shape, 0, dtype=np.float64)
    for i in range(0, a.shape[0]):
        curr = 0 if isnan(a[i]) else a[i]

        if curr == 0:
            out[i] = 0
        elif curr > 0:
            if last <= 0:
                out[i] = curr
            else:
                out[i] = curr + last
        elif curr < 0:
            if last >= 0:
                out[i] = curr
            else:
                out[i] = curr + last

        last = out[i]
    return out


@jit(nopython=True, nogil=True, cache=True)
def _sum_split_by(x1, x2, window=10, n=2):
    out1 = np.full(x1.shape[0], np.nan, dtype=np.float64)
    out2 = np.full(x1.shape[0], np.nan, dtype=np.float64)
    if len(x1) < window:
        return out1, out2
    a1 = sliding_window_view(x1, window)
    a2 = sliding_window_view(x2, window)
    for i, (v1, v2) in enumerate(zip(a1, a2)):
        # 排序两次，解决nan的问题
        b1 = np.argsort(v2)[:n]
        b2 = np.argsort(-v2)[:n]
        out1[i + window - 1] = np.sum(v1[b1])
        out2[i + window - 1] = np.sum(v1[b2])
    return out1, out2


@jit(float64[:](boolean[:], boolean[:], boolean[:], boolean[:], boolean, boolean),
     nopython=True, nogil=True, cache=True)
def _signals_to_size(is_long_entry: np.ndarray, is_long_exit: np.ndarray,
                     is_short_entry: np.ndarray, is_short_exit: np.ndarray,
                     accumulate: bool = False,
                     action: bool = False) -> np.ndarray:
    """将4路信号转换成持仓状态。适合按资产分组后的长表,参考于`vectorbt`

    在`LongOnly`场景下，`is_short_entry`和`is_short_exit`输入数据值都为`False`即可

    Parameters
    ----------
    is_long_entry: np.ndarray
        是否多头入场
    is_long_exit: np.ndarray
        是否多头出场
    is_short_entry: np.ndarray
        是否空头入场
    is_short_exit: np.ndarray
        是否空头出场
    accumulate: bool
        遇到重复信号时是否累计
    action: bool
        返回持仓状态还是下单操作

    Returns
    -------
    np.ndarray
        持仓状态

    Examples
    --------
    ```python
    long_entry = np.array([True, True, False, False, False])
    long_exit = np.array([False, False, True, False, False])
    short_entry = np.array([False, False, True, False, False])
    short_exit = np.array([False, False, False, True, False])

    amount = signals_to_amount(long_entry, long_exit, short_entry, short_exit, accumulate=True, action=False)
    ```

    """
    _amount: float = 0.0  # 持仓状态
    _action: float = 0.0  # 下单方向
    out = np.zeros(len(is_long_entry), dtype=np.float64)
    for i in range(len(is_long_entry)):
        if _amount == 0.0:
            # 多头信号优先级高于空头信号
            if is_long_entry[i]:
                _amount += 1.0
                _action = 1.0
            elif is_short_entry[i]:
                _amount -= 1.0
                _action = -1.0
        elif _amount > 0.0:
            if is_long_exit[i]:
                _amount -= 1.0
                _action = -1.0
            elif is_long_entry[i] and accumulate:
                _amount += 1.0
                _action = 1.0
        else:
            if is_short_exit[i]:
                _amount += 1.0
                _action = 1.0
            elif is_short_entry[i] and accumulate:
                _amount -= 1.0
                _action = -1.0

        out[i] = _action if action else _amount
    return out


@jit(nopython=True, nogil=True, cache=True)
def roll_decay_linear(x1, window, min_periods):
    weights = np.arange(1., window + 1)

    out1 = full_with_window_size(x1, np.nan, dtype=np.float64, window_size=window)
    a1 = sliding_window_with_min_periods(out1, window, min_periods)
    out1[:] = np.nan
    for i, v1 in enumerate(a1):
        if np.isnan(v1).all():
            continue
        mask = ~np.isnan(v1)
        out1[i] = np.average(v1[mask], weights=weights[mask])
    return out1[:x1.shape[0]]


@jit(nopython=True, nogil=True, cache=True)
def roll_decay_exp_window(x1, window, min_periods, factor):
    weights = factor ** np.arange(window - 1, -1, -1)

    out1 = full_with_window_size(x1, np.nan, dtype=np.float64, window_size=window)
    a1 = sliding_window_with_min_periods(out1, window, min_periods)
    out1[:] = np.nan
    for i, v1 in enumerate(a1):
        if np.isnan(v1).all():
            continue
        mask = ~np.isnan(v1)
        out1[i] = np.average(v1[mask], weights=weights[mask])
    return out1[:x1.shape[0]]
