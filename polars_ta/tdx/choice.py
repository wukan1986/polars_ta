import polars as pl

from polars_ta.wq.logical import if_else


def IF(condition: pl.Expr, a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """IF(X,A,B)若X不为0则返回A,否则返回B"""
    return if_else(condition.cast(pl.Boolean), a, b)


def IFN(condition: pl.Expr, a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """IFN(X,A,B)若X不为0则返回B,否则返回A"""
    return if_else(condition.cast(pl.Boolean), b, a)


def VALUEWHEN(condition: pl.Expr, x: pl.Expr) -> pl.Expr:
    return pl.when(condition).then(x).otherwise(None).forward_fill()


IFF = IF
