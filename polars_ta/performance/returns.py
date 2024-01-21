from polars import Expr

from polars_ta.wq.arithmetic import log1p, expm1  # noqa
# 对数收益
from polars_ta.wq.time_series import ts_log_diff as ts_log_return  # noqa
# 简单收益
from polars_ta.wq.time_series import ts_returns as ts_percent_return  # noqa


def ts_cum_return(close: Expr) -> Expr:
    """累计收益"""

    return close / close.drop_nulls().first()


def simple_to_log_return(x: Expr) -> Expr:
    """简单收益率 转 对数收益率"""
    return log1p(x)


def log_to_simple_return(x: Expr) -> Expr:
    """对数收益率 转 简单收益率"""
    return expm1(x)
