from polars import Expr
from polars import Series

from polars_ta.wq.time_series import ts_corr as RELATE  # noqa
from polars_ta.wq.time_series import ts_covariance as COVAR  # noqa
from polars_ta.wq.time_series import ts_std_dev as _ts_std_dev


def _avedev(x: Series) -> Series:
    # 可惜rolling_map后这里已经由Expr变成了Series
    return (x - x.mean()).abs().mean()


def AVEDEV(close: Expr, timeperiod: int = 5) -> Expr:
    """平均绝对偏差"""
    return close.rolling_map(_avedev, timeperiod)


def DEVSQ(close: Expr, timeperiod: int = 5) -> Expr:
    raise


def SLOPE(close: Expr, timeperiod: int = 5) -> Expr:
    raise


def STD(close: Expr, timeperiod: int = 5) -> Expr:
    """估算标准差"""
    return _ts_std_dev(close, timeperiod, 1)


def STDDEV(close: Expr, timeperiod: int = 5) -> Expr:
    """标准偏差?"""
    raise


def STDP(close: Expr, timeperiod: int = 5) -> Expr:
    """总体标准差"""
    return _ts_std_dev(close, timeperiod, 0)


def VAR(close: Expr, timeperiod: int = 5) -> Expr:
    return close.rolling_var(timeperiod, ddof=1)


def VARP(close: Expr, timeperiod: int = 5) -> Expr:
    return close.rolling_var(timeperiod, ddof=0)
