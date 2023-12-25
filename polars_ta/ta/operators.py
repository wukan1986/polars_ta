import numpy as np
import polars as pl

from polars_ta.wq.time_series import ts_max
from polars_ta.wq.time_series import ts_min
from polars_ta.wq.time_series import ts_sum


def ADD(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    return high + low


def DIV(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    return high / low


def MAX(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return ts_max(close, timeperiod)


def MAXINDEX(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """与ts_arg_max的区别是，标记了每个区间最大值的绝对位置，可用来画图标记"""
    return (close.cum_count() + 1 - timeperiod + close.rolling_map(np.argmax, timeperiod)).fill_null(0)


def MIN(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return ts_min(close, timeperiod)


def MAXINDEX(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return (close.cum_count() + 1 - timeperiod + close.rolling_map(np.argmin, timeperiod)).fill_null(0)


def MULT(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    return high * low


def SUB(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    return high - low


def SUM(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return ts_sum(close, timeperiod)
