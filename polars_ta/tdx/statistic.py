import polars as pl

from polars_ta.wq.time_series import ts_corr as RELATE
from polars_ta.wq.time_series import ts_covariance as COVAR
from polars_ta.wq.time_series import ts_std_dev

_ = RELATE, COVAR


def STD(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    """估算标准差"""
    return ts_std_dev(close, timeperiod, 1)


def STDP(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    """总体标准差"""
    return ts_std_dev(close, timeperiod, 0)


def STDDEV(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    """标准偏差?"""
    raise


def VAR(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    return close.rolling_var(timeperiod, ddof=1)


def VARP(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    return close.rolling_var(timeperiod, ddof=0)


def DEVSQ(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    raise


def AVEDEV(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    raise


def SLOPE(close: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    raise
