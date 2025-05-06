import numpy as np
from numba import jit
from numpy import full


@jit(nopython=True, nogil=True, cache=True)
def _triple_barrier(close: np.ndarray, high: np.ndarray, low: np.ndarray, window: int, take_profit: float, stop_loss: float) -> np.ndarray:
    """三重障碍打标法"""
    out = full(close.shape[0], np.nan, dtype=np.float64)
    for i in range(close.shape[0] - window + 1):
        entry_price = close[i]
        if np.isnan(entry_price):
            # out[i] = 0
            continue
        upper_barrier = entry_price * (1 + take_profit)
        lower_barrier = entry_price * (1 - stop_loss)
        for j in range(i + 1, i + window):
            hit_upper = high[j] >= upper_barrier
            hit_lower = low[j] <= lower_barrier
            if hit_upper and hit_lower:
                # TODO 同一天无法知道是先触发止损还是先触发止盈
                # 1. 假定离收盘价远的先触发
                if high[j] - close[j] > close[j] - low[j]:
                    out[i] = 1  # 最高价更远，触发止盈
                else:
                    out[i] = -1  # 最低价更远，触发止损

                # out[i] = -1 # 2. 简化处理认为先触发止损
                break
            if hit_upper:
                out[i] = 1  # 止盈
                break
            if hit_lower:
                out[i] = -1  # 止损
                break
        else:
            # out[i] = 0  # 1. 时间到了触发平仓
            out[i] = np.sign(close[j] / entry_price - 1)  # 2. 时间到了触发平仓

    return out
