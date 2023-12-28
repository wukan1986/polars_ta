from polars import Expr

from polars_ta.wq.time_series import ts_corr
from polars_ta.wq.time_series import ts_std_dev


def BETA(high: Expr, low: Expr, timeperiod: int = 5) -> Expr:
    raise


def CORREL(high: Expr, low: Expr, timeperiod: int = 30) -> Expr:
    return ts_corr(high, low, timeperiod, 1)


def LINEARREG(close: Expr, timeperiod: int = 14) -> Expr:
    raise


def LINEARREG_ANGLE(close: Expr, timeperiod: int = 14) -> Expr:
    raise


def LINEARREG_INTERCEPT(close: Expr, timeperiod: int = 14) -> Expr:
    raise


def LINEARREG_SLOPE(close: Expr, timeperiod: int = 14) -> Expr:
    raise


def STDDEV(close: Expr, timeperiod: int = 5, nbdev: float = 1) -> Expr:
    return ts_std_dev(close, timeperiod, ddof=0) * nbdev


def TSF(close: Expr, timeperiod: int = 14) -> Expr:
    raise


def VAR(close: Expr, timeperiod: int = 5, nbdev: float = 1) -> Expr:
    return close.rolling_var(timeperiod, ddof=0) * nbdev
