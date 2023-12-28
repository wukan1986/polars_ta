from polars import Expr


def AVGPRICE(open: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    return (open + high + low + close) / 4


def MEDPRICE(high: Expr, low: Expr) -> Expr:
    return (high + low) / 2


def TYPPRICE(high: Expr, low: Expr, close: Expr) -> Expr:
    return (high + low + close) / 3


def WCLPRICE(high: Expr, low: Expr, close: Expr) -> Expr:
    return (high + low + close * 2) / 4
