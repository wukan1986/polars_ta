from polars import Series, Expr


def _avedev(x: Series) -> Series:
    # x is Series rather than Expr. Making this function slow
    # 可惜rolling_map后这里已经由Expr变成了Series
    return (x - x.mean()).abs().mean()


def AVEDEV(close: Expr, timeperiod: int = 5) -> Expr:
    """Avoid using this function, it is slow.
    平均绝对偏差
    """
    return close.rolling_map(_avedev, timeperiod)
