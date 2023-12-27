import polars as pl


def AVGPRICE(open: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    return (open + high + low + close) / 4


def MEDPRICE(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    return (high + low) / 2


def TYPPRICE(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    return (high + low + close) / 3


def WCLPRICE(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    return (high + low + close * 2) / 4
