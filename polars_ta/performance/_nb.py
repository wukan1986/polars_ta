import numpy as np
from numba import jit, prange


@jit(nopython=True, nogil=True, fastmath=True)
def np_apply_along_axis(func1d, axis, arr, out):
    if axis == 0:
        for i in range(len(out)):
            out[i] = func1d(arr[:, i])
    else:
        for i in range(len(out)):
            out[i] = func1d(arr[i, :])
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def np_mean(arr, axis, out):
    return np_apply_along_axis(np.mean, axis, arr, out)


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def np_sum(arr, axis, out):
    # sum支持axis
    return np_apply_along_axis(np.sum, axis, arr, out)


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def np_tile(arr, reps):
    m, n = arr.shape
    out = np.empty(shape=(m, n * reps), dtype=arr.dtype)
    for i in range(reps):
        out[:, i * n:(i + 1) * n] = arr
    return out


@jit(nopython=True, nogil=True, fastmath=True, cache=True, parallel=True)
def _sub_portfolio_returns(m: int, n: int, weights: np.ndarray, returns: np.ndarray, period: int = 3, is_mean: bool = True) -> np.ndarray:
    # tile时可能长度不够，所以补充一段
    weights = np.concatenate((weights, weights[:period]), axis=0)
    # 记录每份的收益率
    out = np.zeros(shape=(m, period), dtype=float)
    # 资金分成period份
    for i in prange(period):
        # 某一天的持仓需要持续period天
        w = np_tile(weights[i::period], period).reshape(-1, n)
        if i > 0:
            # shift操作
            w[i:] = w[:-i]
            w[:i] = 0
        w = w[:m]

        if n == 1:
            # 一条资产，直接返回
            out[:, i] = (returns * w).flatten()
            continue

        # 计算此份资金的收益，净值从1开始
        if is_mean:
            # 等权分配
            np_mean(returns * w, axis=1, out=out[:, i])
        else:
            # 将权重分配交给外部。一般pos.abs.sum==1
            np_sum(returns * w, axis=1, out=out[:, i])
    return out
