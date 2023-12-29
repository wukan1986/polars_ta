from polars import Expr
from polars import Int32, Boolean

from polars_ta import TA_EPSILON
from polars_ta.tdx.reference import SUM


def CROSS(a: Expr, b: Expr) -> Expr:
    return (a <= (b - TA_EPSILON)).shift(1) & (a > (b + TA_EPSILON))


def DOWNNDAY(close: Expr, N: int) -> Expr:
    return NDAY(close.shift(), close, N)


def EVERY(condition: Expr, N: int) -> Expr:
    return SUM(condition.cast(Int32), N) == N


def EXIST(condition: Expr, N: int) -> Expr:
    return SUM(condition.cast(Int32), N) > 0


def EXISTR(condition: Expr, a: int, b: int) -> Expr:
    return EXIST(condition, a - b).shift(b)


def LAST(condition: Expr, a: int, b: int) -> Expr:
    return EVERY(condition, a - b).shift(b)


def LONGCROSS(a: Expr, b: Expr, N: int) -> Expr:
    return CROSS(a, b) & EVERY(a < (b - TA_EPSILON), N).shift(1)


def NDAY(close: Expr, open_: Expr, N: int) -> Expr:
    return EVERY(close > (open_ + TA_EPSILON), N)


def NOT(condition: Expr) -> Expr:
    return ~condition.cast(Boolean)


def UPNDAY(close: Expr, N: int) -> Expr:
    return NDAY(close, close.shift(), N)


# 东方财富中有此两函数
ALL = EVERY
ANY = EXIST
