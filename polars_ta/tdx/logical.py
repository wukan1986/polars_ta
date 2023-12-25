import polars as pl

from polars_ta import TA_EPSILON
from polars_ta.tdx.reference import SUM


def CROSS(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return (a <= b - TA_EPSILON).shift(1) & (a > b + TA_EPSILON)


def DOWNNDAY(close: pl.Expr, timeperiod: int) -> pl.Expr:
    return NDAY(close.shift(), close, timeperiod)


def EVERY(x: pl.Expr, timeperiod: int) -> pl.Expr:
    return SUM(x.cast(pl.Int32), timeperiod) == timeperiod


def EXIST(x: pl.Expr, timeperiod: int) -> pl.Expr:
    return SUM(x.cast(pl.Int32), timeperiod) > 0


def EXISTR(x: pl.Expr, a: int, b: int) -> pl.Expr:
    return EXIST(x, a - b).shift(b)


def LAST(x: pl.Expr, a: int, b: int) -> pl.Expr:
    return EVERY(x, a - b).shift(b)


def LONGCROSS(a: pl.Expr, b: pl.Expr, n: int) -> pl.Expr:
    return CROSS(a, b) & EVERY(a < b - TA_EPSILON, n).shift(1)


def NDAY(close: pl.Expr, open_: pl.Expr, timeperiod: int) -> pl.Expr:
    return EVERY(close > open_ + TA_EPSILON, timeperiod)


def NOT(x: pl.Expr) -> pl.Expr:
    return ~x


def UPNDAY(close: pl.Expr, timeperiod: int) -> pl.Expr:
    return NDAY(close, close.shift(), timeperiod)


# 东方财富中有此两函数
ALL = EVERY
ANY = EXIST
