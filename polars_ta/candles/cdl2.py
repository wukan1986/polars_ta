"""
In this file we use
open_: Expr, high: Expr, low: Expr, close: Expr
for all parameters
to ensure the similar function signatures when calling them

本文件中参数全用
open_: Expr, high: Expr, low: Expr, close: Expr
统一的好处是在使用时不用考虑函数调用区别
"""
from polars import Expr

from polars_ta.candles.cdl1 import lower_body, upper_body


# https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_utility.h#L360
def ts_gap_up(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """跳空高开"""
    return low > high.shift(1)


def ts_gap_down(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """跳空低开"""
    return high < low.shift(1)


def ts_real_body_gap_up(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """实体跳空高开"""
    return lower_body(open_, high, low, close) > upper_body(open_, high, low, close).shift(1)


def ts_real_body_gap_down(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """实体跳空低开"""
    return upper_body(open_, high, low, close) < lower_body(open_, high, low, close).shift(1)
