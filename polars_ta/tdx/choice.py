import polars as pl

from polars_ta.wq.logical import if_else


def IF(x: pl.Expr, a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """IF(X,A,B)若X不为0则返回A,否则返回B"""
    return if_else(x.cast(pl.Boolean), a, b)


def IFN(x: pl.Expr, a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """IFN(X,A,B)若X不为0则返回B,否则返回A"""
    return if_else(x.cast(pl.Boolean), b, a)


IFF = IF
