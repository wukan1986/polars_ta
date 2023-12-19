import polars as pl

from polars_ta.ta.overlap import SMA, EMA
from polars_ta.wq.time_series import ts_delta, ts_returns


def MACD_macd(close: pl.Expr, fastperiod: int = 12, slowperiod: int = 26) -> pl.Expr:
    """

    Notes
    -----
    talib.MACD有效数据按fastperiod，而本项目按slowperiod

    """
    return EMA(close, fastperiod) - EMA(close, slowperiod)


def MACD_macdhist(close: pl.Expr, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pl.Expr:
    """

    Notes
    -----
    中国版多了乘2

    """
    macd = MACD_macd(close, fastperiod, slowperiod)
    signal = EMA(macd, signalperiod)
    return macd - signal


def MACD_macdsignal(close: pl.Expr, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pl.Expr:
    macd = MACD_macd(close, fastperiod, slowperiod)
    signal = EMA(macd, signalperiod)
    return signal


def MOM(close: pl.Expr, timeperiod: int = 10) -> pl.Expr:
    return ts_delta(close, timeperiod)


def ROCP(close: pl.Expr, timeperiod: int = 10) -> pl.Expr:
    return close / close.shift(timeperiod)


def ROCR(close: pl.Expr, timeperiod: int = 10) -> pl.Expr:
    return ts_returns(close, timeperiod)


def STOCHF_fastd(high: pl.Expr, low: pl.Expr, close: pl.Expr, fastk_period: int = 5, fastd_period: int = 3) -> pl.Expr:
    return SMA(STOCHF_fastk(high, low, close, fastk_period), fastd_period)


def STOCHF_fastk(high: pl.Expr, low: pl.Expr, close: pl.Expr, fastk_period: int = 5) -> pl.Expr:
    """

    Notes
    -----
    talib.STOCHF版相当于多乘了100

    """
    return (close - low.rolling_min(fastk_period)) / (high.rolling_max(fastk_period) - low.rolling_min(fastk_period))


def TRIX(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    EMA1 = EMA(close, timeperiod)
    EMA2 = EMA(EMA1, timeperiod)
    EMA3 = EMA(EMA2, timeperiod)
    return ROCP(EMA3, 1)


def WILLR(high: pl.Expr, low: pl.Expr, close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    """

    Notes
    -----
    talib.WILLR版相当于多乘了-100，但个人认为没有必要

    References
    ----------
    https://www.investopedia.com/terms/w/williamsr.asp
    https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r

    """
    return (high.rolling_max(timeperiod) - close) / (high.rolling_max(timeperiod) - low.rolling_min(timeperiod))
