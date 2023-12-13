import polars as pl


def log_return(close: pl.Expr, timeperiod: int = 1) -> pl.Expr:
    """对数收益

    Notes
    -----
    对数收益可以在时序上累计求和，但不能在横截面上求平均
    TODO  是否可以求几何平均？
    """

    return close.log().diff(timeperiod)


def percent_return(close: pl.Expr, timeperiod: int = 1) -> pl.Expr:
    """百分比收益"""

    return close.pct_change(timeperiod)


def cum_return(close: pl.Expr) -> pl.Expr:
    """累计收益"""

    return close / close.drop_nulls().first()
