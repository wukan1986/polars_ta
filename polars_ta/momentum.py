import polars as pl

from polars_ta.overlap import SMA, EMA


def MOM(close: pl.Expr, timeperiod: int = 10) -> pl.Expr:
    return close.diff(timeperiod)


def ROCR(close: pl.Expr, timeperiod: int = 10) -> pl.Expr:
    return close / close.shift(timeperiod)


def ROCP(close: pl.Expr, timeperiod: int = 10) -> pl.Expr:
    """

    Notes
    -----
    ROCR(close, timeperiod) - 1

    """
    return close.pct_change(timeperiod)


def WILLR(high: pl.Expr, low: pl.Expr, close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    """

    Notes
    -----
    talib.WILLR版相当于多乘了-100

    """
    return (high.rolling_max(timeperiod) - close) / (high.rolling_max(timeperiod) - low.rolling_min(timeperiod))


def STOCHF_fastk(high: pl.Expr, low: pl.Expr, close: pl.Expr, fastk_period: int = 5) -> pl.Expr:
    """

    Notes
    -----
    talib.STOCHF版相当于多乘了100

    """
    return (close - low.rolling_min(fastk_period)) / (high.rolling_max(fastk_period) - low.rolling_min(fastk_period))


def STOCHF_fastd(high: pl.Expr, low: pl.Expr, close: pl.Expr, fastk_period: int = 5, fastd_period: int = 3) -> pl.Expr:
    return SMA(STOCHF_fastk(high, low, close, fastk_period), fastd_period)


def MACD_macd(close: pl.Expr, fastperiod: int = 12, slowperiod: int = 26) -> pl.Expr:
    """

    Notes
    -----
    talib.MACD有效数据按fastperiod，而本项目按slowperiod

    """
    return EMA(close, fastperiod) - EMA(close, slowperiod)


def MACD_macdsignal(close: pl.Expr, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pl.Expr:
    macd = MACD_macd(close, fastperiod, slowperiod)
    signal = EMA(macd, signalperiod)
    return signal


def MACD_macdhist(close: pl.Expr, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pl.Expr:
    """

    Notes
    -----
    中国版多了乘2

    """
    macd = MACD_macd(close, fastperiod, slowperiod)
    signal = EMA(macd, signalperiod)
    return macd - signal
