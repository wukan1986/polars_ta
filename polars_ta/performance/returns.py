from polars import Expr

from polars_ta.wq.arithmetic import log1p, expm1


def ts_cum_return(close: Expr) -> Expr:
    """
    cumulative return
    累计收益
    """

    return close / close.drop_nulls().first()


def simple_to_log_return(x: Expr) -> Expr:
    """简单收益率 转 对数收益率"""
    return log1p(x)


def log_to_simple_return(x: Expr) -> Expr:
    """对数收益率 转 简单收益率"""
    return expm1(x)
