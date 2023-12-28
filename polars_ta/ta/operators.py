from polars import Expr

from polars_ta.wq.time_series import ts_arg_max
from polars_ta.wq.time_series import ts_arg_min
from polars_ta.wq.time_series import ts_max
from polars_ta.wq.time_series import ts_min
from polars_ta.wq.time_series import ts_sum


def ADD(high: Expr, low: Expr) -> Expr:
    return high + low


def DIV(high: Expr, low: Expr) -> Expr:
    return high / low


def MAX(close: Expr, timeperiod: int = 30) -> Expr:
    """

    Notes
    -----
    时序上窗口最大，不要与多列最大搞混

    """
    return ts_max(close, timeperiod)


def MAXINDEX(close: Expr, timeperiod: int = 30) -> Expr:
    """与ts_arg_max的区别是，标记了每个区间最大值的绝对位置，可用来画图标记"""
    a = close.cum_count()
    b = ts_arg_max(close, timeperiod)
    return a - b


def MIN(close: Expr, timeperiod: int = 30) -> Expr:
    return ts_min(close, timeperiod)


def MININDEX(close: Expr, timeperiod: int = 30) -> Expr:
    a = close.cum_count()
    b = ts_arg_min(close, timeperiod)
    return a - b


def MULT(high: Expr, low: Expr) -> Expr:
    return high * low


def SUB(high: Expr, low: Expr) -> Expr:
    return high - low


def SUM(close: Expr, timeperiod: int = 30) -> Expr:
    return ts_sum(close, timeperiod)
