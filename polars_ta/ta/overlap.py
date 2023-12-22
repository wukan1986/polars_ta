from math import ceil, floor

import polars as pl

from polars_ta.ta.statistic import STDDEV
from polars_ta.wq.time_series import ts_mean, ts_decay_linear


def BBANDS_upperband(close: pl.Expr, timeperiod: int = 5, nbdevup: float = 2) -> pl.Expr:
    """布林线上轨

    Notes
    -----
    1. 想替换中线算法时，参考此代码新建一个函数
    2. 想生成下轨时，nbdevup用负数
    """
    return SMA(close, timeperiod) + STDDEV(close, timeperiod, ddof=0) * nbdevup


def DEMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    EMA1 = EMA(close, timeperiod)
    EMA2 = EMA(EMA1, timeperiod)
    return EMA1 * 2 - EMA2


def EMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """

    References
    ----------
    https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.ewm_mean.html#polars.Expr.ewm_mean

    """
    # 相当于alpha=2/(1+timeperiod)
    return close.ewm_mean(span=timeperiod, adjust=False, min_periods=timeperiod)


def MIDPOINT(close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    return (close.rolling_max(timeperiod) + close.rolling_min(timeperiod)) / 2


def MIDPRICE(high: pl.Expr, low: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    return (high.rolling_max(timeperiod) + low.rolling_min(timeperiod)) / 2


def RMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """TA-Lib没有明确的提供此算法，这里只是为了调用方便而放在此处

    References
    ----------
    https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.ewm_mean.html#polars.Expr.ewm_mean

    """
    return close.ewm_mean(alpha=1 / timeperiod, adjust=False, min_periods=timeperiod)


def SMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return ts_mean(close, timeperiod)


def TEMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """

    Notes
    -----
    嵌套层数过多，也许直接调用talib.TEMA更快

    """
    EMA1 = EMA(close, timeperiod)
    EMA2 = EMA(EMA1, timeperiod)
    EMA3 = EMA(EMA2, timeperiod)
    return (EMA1 - EMA2) * 3 + EMA3


def TRIMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    SMA1 = SMA(close, ceil(timeperiod / 2))
    return SMA(SMA1, floor(timeperiod / 2) + 1)


def WMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return ts_decay_linear(close, timeperiod)
