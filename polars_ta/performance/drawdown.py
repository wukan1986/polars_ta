import polars as pl


def drawdown(x: pl.Expr) -> pl.Expr:
    return x.cum_max() - x
