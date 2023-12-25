import polars as pl

from polars_ta.wq.time_series import ts_corr as RELATE  # noqa
from polars_ta.wq.time_series import ts_covariance as COVAR  # noqa
from polars_ta.wq.time_series import ts_std_dev


def _avedev(x: pl.Series) -> pl.Series:
    # 可惜rolling_map后这里已经由Expr变成了Series
    return (x - x.mean()).abs().mean()


def AVEDEV(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    """平均绝对偏差"""
    return close.rolling_map(_avedev, timeperiod)


def DEVSQ(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    raise


def SLOPE(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    raise


def STD(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    """估算标准差"""
    return ts_std_dev(close, timeperiod, 1)


def STDDEV(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    """标准偏差?"""
    raise


def STDP(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    """总体标准差"""
    return ts_std_dev(close, timeperiod, 0)


def VAR(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    return close.rolling_var(timeperiod, ddof=1)


def VARP(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    return close.rolling_var(timeperiod, ddof=0)
