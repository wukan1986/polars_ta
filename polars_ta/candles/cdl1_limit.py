"""
In this file we use
open_: Expr, high: Expr, low: Expr, close: Expr, high_limit: Expr
open_: Expr, high: Expr, low: Expr, close: Expr, low_limit: Expr
for all parameters
to ensure the similar function signatures when calling them

本文件中参数全用
open_: Expr, high: Expr, low: Expr, close: Expr, high_limit: Expr
open_: Expr, high: Expr, low: Expr, close: Expr, low_limit: Expr
统一的好处是在使用时不用考虑函数调用区别
"""
from polars import Expr

from polars_ta import TA_EPSILON
from polars_ta.candles.cdl1 import four_price_doji, dragonfly, gravestone


def limit_up(price: Expr, high_limit: Expr) -> Expr:
    """
    unified handles for
    reached `high_limit` at `open_`
    reached `high_limit` at `close`
    reached `high_limit` at `high` but `close` < `high_limit`  (at least once reached `high_limit` in this day)
    开盘 涨停
    收盘 涨停
    最高 曾涨停
    """
    return price >= high_limit - TA_EPSILON


def limit_up_at_open(open_: Expr, high: Expr, low: Expr, close: Expr, high_limit: Expr) -> Expr:
    """
    reached `high_limit` at `open_`
    开盘 涨停
    """
    return limit_up(open_, high_limit)


def limit_up_at_close(open_: Expr, high: Expr, low: Expr, close: Expr, high_limit: Expr) -> Expr:
    """
    reached `high_limit` at `close`
    收盘 涨停
    """
    return limit_up(close, high_limit)


def limit_up_at_high(open_: Expr, high: Expr, low: Expr, close: Expr, high_limit: Expr) -> Expr:
    """
    reached `high_limit` at `high` but `close` < `high_limit`  (at least once reached `high_limit` in this day)
    曾涨停
    """
    return limit_up(high, high_limit) & ~limit_up(close, high_limit)


def limit_up_four_price_doji(open_: Expr, high: Expr, low: Expr, close: Expr, high_limit: Expr) -> Expr:
    """
    `low` = `high` = `open_` = `close` = `high_limit`
    一字涨停
    """
    return four_price_doji(open_, high, low, close) & limit_up(close, high_limit)


def limit_up_dragonfly(open_: Expr, high: Expr, low: Expr, close: Expr, high_limit: Expr) -> Expr:
    """
    `low` < `high` = `open_` = `close` = `high_limit`
    T字涨停"""
    return limit_up(close, high_limit) & dragonfly(open_, high, low, close)


# ======================================
def limit_down(price: Expr, low_limit: Expr) -> Expr:
    """
    unified handles for
    reached `low_limit` at `open_`
    reached `low_limit` at `close`
    reached `low_limit` at `high` but `close` > `low_limit`  (at least once reached `low_limit` in this day)
    开盘 跌停
    收盘 跌停
    最低 曾跌停
    """
    return price <= low_limit + TA_EPSILON


def limit_down_at_open(open_: Expr, high: Expr, low: Expr, close: Expr, low_limit: Expr) -> Expr:
    """
    reached `low_limit` at `open_`
    开盘 跌停
    """
    return limit_down(open_, low_limit)


def limit_down_at_close(open_: Expr, high: Expr, low: Expr, close: Expr, low_limit: Expr) -> Expr:
    """
    reached `low_limit` at `close`
    收盘 跌停
    """
    return limit_down(close, low_limit)


def limit_down_at_high(open_: Expr, high: Expr, low: Expr, close: Expr, low_limit: Expr) -> Expr:
    """
    reached `low_limit` at `high` but `close` > `low_limit`  (at least once reached `low_limit` in this day)
    曾跌停
    """
    return limit_down(low, low_limit) & ~limit_down(close, low_limit)


def limit_down_four_price_doji(open_: Expr, high: Expr, low: Expr, close: Expr, low_limit: Expr) -> Expr:
    """
    `low` = `high` = `open_` = `close` = `low_limit`
    一字跌停
    """
    return four_price_doji(open_, high, low, close) & limit_down(close, low_limit)


def limit_down_gravestone(open_: Expr, high: Expr, low: Expr, close: Expr, low_limit: Expr) -> Expr:
    """
    `high` > `low` = `open_` = `close` = `low_limit`
    T字跌停
    """
    return limit_down(open_, low_limit) & gravestone(open_, high, low, close)
