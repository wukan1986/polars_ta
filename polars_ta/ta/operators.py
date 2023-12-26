import polars as pl

from polars_ta.wq.time_series import ts_arg_max
from polars_ta.wq.time_series import ts_arg_min
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
    a = close.cum_count()
    b = ts_arg_max(close, timeperiod)
    return a - b


def MIN(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return ts_min(close, timeperiod)


def MININDEX(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    a = close.cum_count()
    b = ts_arg_min(close, timeperiod)
    return a - b


def MULT(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    return high * low


def SUB(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    return high - low


def SUM(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return ts_sum(close, timeperiod)
