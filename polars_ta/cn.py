"""
以下指标由于国内外差异比较大，所以另提供了一份国内版
"""
import polars as pl

from polars_ta.ta.momentum import MACD_macdhist


def MACD(close: pl.Expr, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pl.Expr:
    return MACD_macdhist(close, fastperiod, slowperiod, signalperiod) * 2
