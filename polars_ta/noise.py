"""
The following comes from Trading Systems and Methods, Chapter 1, Measuring Noise

We do not provide ATR and STD here.


ATR与STD也是一种度量波动的方法，这里不再提供

以下方法来自于Trading Systems and Methods, Chapter 1, Measuring Noise

References
----------
https://zhuanlan.zhihu.com/p/544744582

"""
import numpy as np
from polars import Expr


def ts_efficiency_ratio(close: Expr, timeperiod: int = 14) -> Expr:
    """效率系数。值越大，噪音越小。最大值为1，最小值为0

    本质上是位移除以路程
    """
    t1 = close.diff(timeperiod).abs()
    t2 = close.diff(1).abs().rolling_sum(timeperiod)
    return t1 / t2


def ts_price_density(high: Expr, low: Expr, timeperiod: int = 14) -> Expr:
    """价格密度。值越大，噪音越大

    如果K线高低相连，上涨为1，下跌也为1
    如果K线高低平行，值大于1，最大为timeperiod
    """
    t1 = (high - low).rolling_sum(timeperiod)
    t2 = high.rolling_max(timeperiod) - low.rolling_min(timeperiod)
    return t1 / t2


def ts_fractal_dimension(high: Expr, low: Expr, close: Expr, timeperiod: int = 14) -> Expr:
    """分形维度。值越大，噪音越大"""
    n1 = (1 / timeperiod) ** 2
    n2 = np.log(2)
    n3 = np.log(2 * timeperiod)

    t1 = high.rolling_max(timeperiod) - low.rolling_min(timeperiod)
    t2 = close.diff(1)

    L = (n1 + t2 / t1).sqrt().rolling_sum(timeperiod)
    return 1 + (L.log() + n2) / n3
