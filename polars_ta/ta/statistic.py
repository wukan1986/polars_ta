import polars as pl

from polars_ta.wq.time_series import ts_corr
from polars_ta.wq.time_series import ts_std_dev


def BETA(high: pl.Expr, low: pl.Expr, timeperiod: int = 5) -> pl.Expr:
    raise


def CORREL(high: pl.Expr, low: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return ts_corr(high, low, timeperiod, 1)


def LINEARREG(close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    raise


def LINEARREG_ANGLE(close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    raise


def LINEARREG_INTERCEPT(close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    raise


def LINEARREG_SLOPE(close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    raise


def STDDEV(close: pl.Expr, timeperiod: int = 5, nbdev: float = 1) -> pl.Expr:
    return ts_std_dev(close, timeperiod, ddof=0) * nbdev


def TSF(close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    raise


def VAR(close: pl.Expr, timeperiod: int = 5, nbdev: float = 1) -> pl.Expr:
    return close.rolling_var(timeperiod, ddof=0) * nbdev
