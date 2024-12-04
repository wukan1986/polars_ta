from polars import Expr


def AVGPRICE(open: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """(open + high + low + close) / 4

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_AVGPRICE.c#L187

    """
    return (open + high + low + close) / 4


def MEDPRICE(high: Expr, low: Expr) -> Expr:
    """(high + low) / 2

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_MEDPRICE.c#L180

    """
    return (high + low) / 2


def TYPPRICE(high: Expr, low: Expr, close: Expr) -> Expr:
    """(high + low + close) / 3

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_TYPPRICE.c#L185

    """
    return (high + low + close) / 3


def WCLPRICE(high: Expr, low: Expr, close: Expr) -> Expr:
    """(high + low + close * 2) / 4

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_WCLPRICE.c#L184

    """
    return (high + low + close * 2) / 4
