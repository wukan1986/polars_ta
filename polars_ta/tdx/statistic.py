from polars import Expr

from polars_ta.tdx._nb import roll_avedev
from polars_ta.utils.numba_ import batches_i1_o1
from polars_ta.wq.time_series import ts_corr as RELATE  # noqa
from polars_ta.wq.time_series import ts_covariance as COVAR  # noqa
from polars_ta.wq.time_series import ts_std_dev as _ts_std_dev


def AVEDEV(close: Expr, timeperiod: int = 5) -> Expr:
    """平均绝对偏差"""
    return close.map_batches(lambda x1: batches_i1_o1(x1.to_numpy(), roll_avedev, timeperiod))


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
