from math import ceil, floor

from polars import Expr

from polars_ta.ta.operators import MAX
from polars_ta.ta.operators import MIN
from polars_ta.ta.statistic import STDDEV
from polars_ta.wq.time_series import ts_decay_linear as WMA  # noqa
from polars_ta.wq.time_series import ts_mean as SMA  # noqa


def BBANDS_upperband(close: Expr, timeperiod: int = 5, nbdevup: float = 2) -> Expr:
    """Bollinger Bands Upper Band
    布林线上轨

    Notes
    -----
    1. You may create a new functino based on this, to replace the middle band
    2. use negative values in `nbdevup` for the lower band

    1. 想替换中线算法时，参考此代码新建一个函数
    2. 想生成下轨时，nbdevup用负数
    """
    return SMA(close, timeperiod) + STDDEV(close, timeperiod, nbdevup)


def DEMA(close: Expr, timeperiod: int = 30) -> Expr:
    EMA1 = EMA(close, timeperiod)
    EMA2 = EMA(EMA1, timeperiod)
    return EMA1 * 2 - EMA2


def EMA(close: Expr, timeperiod: int = 30) -> Expr:
    """

    References
    ----------
    https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.ewm_mean.html#polars.Expr.ewm_mean

    """
    # 相当于alpha=2/(1+timeperiod)
    return close.ewm_mean(span=timeperiod, adjust=False, min_periods=timeperiod)


def KAMA(close: Expr, timeperiod: int = 30) -> Expr:
    raise


def MIDPOINT(close: Expr, timeperiod: int = 14) -> Expr:
    """MIDPOINT = (Highest Value + Lowest Value)/2

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_MIDPOINT.c#L198

    """
    return (MAX(close, timeperiod) + MIN(close, timeperiod)) / 2


def MIDPRICE(high: Expr, low: Expr, timeperiod: int = 14) -> Expr:
    """MIDPRICE = (Highest High + Lowest Low)/2

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_MIDPRICE.c#L202

    """
    return (MAX(high, timeperiod) + MIN(low, timeperiod)) / 2


def RMA(close: Expr, timeperiod: int = 30) -> Expr:
    """TA-Lib does not provide this algorithm explicitly, it is just put here for convenience
    TA-Lib没有明确的提供此算法，这里只是为了调用方便而放在此处

    References
    ----------
    https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.ewm_mean.html#polars.Expr.ewm_mean
    https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/rma.py

    """
    return close.ewm_mean(alpha=1 / timeperiod, adjust=False, min_periods=timeperiod)


def TEMA(close: Expr, timeperiod: int = 30) -> Expr:
    """

    Notes
    -----
    todo: if the nesting level is too deep, maybe call talib.TEMA directly
    嵌套层数过多，也许直接调用talib.TEMA更快

    """
    EMA1 = EMA(close, timeperiod)
    EMA2 = EMA(EMA1, timeperiod)
    EMA3 = EMA(EMA2, timeperiod)
    return (EMA1 - EMA2) * 3 + EMA3


def TRIMA(close: Expr, timeperiod: int = 30) -> Expr:
    SMA1 = SMA(close, ceil(timeperiod / 2))
    return SMA(SMA1, floor(timeperiod / 2) + 1)
