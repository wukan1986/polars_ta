import polars as pl


def STDDEV(close: pl.Expr, timeperiod: int = 5, ddof: int = 0) -> pl.Expr:
    return close.rolling_std(timeperiod, ddof=ddof)


def VAR(close: pl.Expr, timeperiod: int = 5, ddof: int = 0) -> pl.Expr:
    return close.rolling_var(timeperiod, ddof=ddof)
