import polars as pl

from polars_ta.wq.time_series import ts_std_dev

STDDEV = ts_std_dev


def VAR(close: pl.Expr, timeperiod: int = 5, ddof: int = 0) -> pl.Expr:
    return close.rolling_var(timeperiod, ddof=ddof)
