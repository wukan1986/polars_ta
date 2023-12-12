"""
以下指标由于国内外差异比较大，所以另提供了一份国内版
"""
import polars as pl

from polars_ta.momentum import MACD_macdhist
from polars_ta.overlap import SMA
from polars_ta.volatility import TRANGE


def ATR(high: pl.Expr, low: pl.Expr, close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    return SMA(TRANGE(high, low, close), timeperiod)


def MACD(close: pl.Expr, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pl.Expr:
    return MACD_macdhist(close, fastperiod, slowperiod, signalperiod) * 2
