import polars as pl

from polars_ta.wq.time_series import ts_std_dev


def STDDEV(close: pl.Expr, timeperiod: int = 5, ddof: int = 0) -> pl.Expr:
    return ts_std_dev(close, timeperiod, ddof)


def VAR(close: pl.Expr, timeperiod: int = 5, ddof: int = 0) -> pl.Expr:
    return close.rolling_var(timeperiod, ddof=ddof)
