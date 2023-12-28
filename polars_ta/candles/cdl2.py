import polars as pl

from polars_ta.candles.cdl1 import lower_body, upper_body


# https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_utility.h#L360
def gap_up(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    """跳空高开"""
    return low > high.shift(1)


def gap_down(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    """跳空低开"""
    return high < low.shift(1)


def real_body_gap_up(open_: pl.Expr, close: pl.Expr) -> pl.Expr:
    """实体跳空高开"""
    return lower_body(open_, close) > upper_body(open_, close).shift(1)


def real_body_gap_down(open_: pl.Expr, close: pl.Expr) -> pl.Expr:
    """实体跳空低开"""
    return upper_body(open_, close) < lower_body(open_, close).shift(1)
