import polars as pl

from polars_ta.statistic import STDDEV


def SMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return close.rolling_mean(timeperiod)


def WMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return close.rolling_mean(timeperiod, weights=pl.arange(1, timeperiod + 1, eager=True) / 30)


def EMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return close.ewm_mean(span=timeperiod, adjust=False, min_periods=timeperiod)


def BBANDS_upperband(close: pl.Expr, timeperiod: int = 5, nbdevup: float = 2) -> pl.Expr:
    """布林线上轨

    Notes
    -----
    1. 想替换中线算法时，参考此代码新建一个函数
    2. 想生成下轨时，nbdevup用负数
    """
    return SMA(close, timeperiod) + STDDEV(close, timeperiod, ddof=0) * nbdevup


def MIDPOINT(close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    return (close.rolling_max(timeperiod) + close.rolling_min(timeperiod)) / 2


def MIDPRICE(high: pl.Expr, low: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    return (high.rolling_max(timeperiod) + low.rolling_min(timeperiod)) / 2


def DEMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    EMA1 = EMA(close, timeperiod)
    EMA2 = EMA(EMA1, timeperiod)
    return EMA1 * 2 - EMA2


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
