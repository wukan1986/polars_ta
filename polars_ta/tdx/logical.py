from polars import Expr
from polars import Int32, Boolean

from polars_ta import TA_EPSILON
from polars_ta.tdx.reference import SUM


def CROSS(a: Expr, b: Expr) -> Expr:
    return (a <= (b - TA_EPSILON)).shift(1) & (a > (b + TA_EPSILON))


def DOWNNDAY(close: Expr, timeperiod: int) -> Expr:
    return NDAY(close.shift(), close, timeperiod)


def EVERY(condition: Expr, timeperiod: int) -> Expr:
    return SUM(condition.cast(Int32), timeperiod) == timeperiod


def EXIST(condition: Expr, timeperiod: int) -> Expr:
    return SUM(condition.cast(Int32), timeperiod) > 0


def EXISTR(condition: Expr, a: int, b: int) -> Expr:
    return EXIST(condition, a - b).shift(b)


def LAST(condition: Expr, a: int, b: int) -> Expr:
    return EVERY(condition, a - b).shift(b)


def LONGCROSS(a: Expr, b: Expr, n: int) -> Expr:
    return CROSS(a, b) & EVERY(a < (b - TA_EPSILON), n).shift(1)


def NDAY(close: Expr, open_: Expr, timeperiod: int) -> Expr:
    return EVERY(close > (open_ + TA_EPSILON), timeperiod)


def NOT(condition: Expr) -> Expr:
    return ~condition.cast(Boolean)


def UPNDAY(close: Expr, timeperiod: int) -> Expr:
    return NDAY(close, close.shift(), timeperiod)


# 东方财富中有此两函数
ALL = EVERY
ANY = EXIST
