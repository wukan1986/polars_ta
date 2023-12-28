from polars import Boolean
from polars import Expr
from polars import when

from polars_ta.wq.logical import if_else


def IF(condition: Expr, a: Expr, b: Expr) -> Expr:
    """IF(X,A,B)若X不为0则返回A,否则返回B"""
    return if_else(condition.cast(Boolean), a, b)


def IFN(condition: Expr, a: Expr, b: Expr) -> Expr:
    """IFN(X,A,B)若X不为0则返回B,否则返回A"""
    return if_else(condition.cast(Boolean), b, a)


def VALUEWHEN(condition: Expr, x: Expr) -> Expr:
    return when(condition).then(x).otherwise(None).forward_fill()


IFF = IF
