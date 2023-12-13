import polars as pl


def drawdown(close: pl.Expr) -> pl.Expr:
    return close.cum_max() - close
