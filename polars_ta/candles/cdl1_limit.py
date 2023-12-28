from polars import Expr

from polars_ta import TA_EPSILON


def limit_up(ohc: Expr, high_limit: Expr) -> Expr:
    """
    开盘 涨停
    收盘 涨停
    最高 曾涨停
    """
    return ohc >= high_limit - TA_EPSILON


def limit_down(olc: Expr, low_limit: Expr) -> Expr:
    """
    开盘 跌停
    收盘 跌停
    最低 曾跌停
    """
    return olc <= low_limit + TA_EPSILON


def limit_up_not_close(high: Expr, close: Expr, high_limit: Expr) -> Expr:
    """曾涨停"""
    return limit_up(high, high_limit) & ~limit_up(close, high_limit)


def limit_down_not_close(low: Expr, close: Expr, low_limit: Expr) -> Expr:
    """曾跌停"""
    return limit_down(low, low_limit) & ~limit_down(close, low_limit)


def line(high: Expr, low: Expr) -> Expr:
    """一字"""
    return low > (high - TA_EPSILON)


def cross(open_: Expr, close: Expr) -> Expr:
    """十字(含一字、T字)"""
    return (open_ - close).abs() < TA_EPSILON


def limit_up_line(high: Expr, low: Expr, close: Expr, high_limit: Expr) -> Expr:
    """一字涨停"""
    return line(high, low) & limit_up(close, high_limit)


def limit_down_line(high: Expr, low: Expr, close: Expr, low_limit: Expr) -> Expr:
    """一字跌停"""
    return line(high, low) & limit_down(close, low_limit)


def limit_up_t(open_: Expr, high: Expr, low: Expr, close: Expr, high_limit: Expr) -> Expr:
    """T字涨停"""
    return limit_up(open_, high_limit) & limit_up(close, high_limit) & ~line(high, low)


def limit_down_t(open_: Expr, high: Expr, low: Expr, close: Expr, low_limit: Expr) -> Expr:
    """T字跌停"""
    return limit_down(open_, low_limit) & limit_down(close, low_limit) & ~line(high, low)
