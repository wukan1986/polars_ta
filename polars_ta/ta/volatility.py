from polars import Expr, max_horizontal

from polars_ta.ta.overlap import RMA


def TRANGE(high: Expr, low: Expr, close: Expr) -> Expr:
    """

    Notes
    -----
    the 0-th position is `x` rather than `nan` in talib

    第0位置为max(x,nan,nan)=x,比talib多一个值

    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return max_horizontal(tr1, tr2, tr3)


def ATR(high: Expr, low: Expr, close: Expr, timeperiod: int = 14) -> Expr:
    """"""
    return RMA(TRANGE(high, low, close), timeperiod)


def NATR(high: Expr, low: Expr, close: Expr, timeperiod: int = 14) -> Expr:
    """

    Notes
    -----
    talib.ATR multiples another 100

    talib.ATR版相当于多乘了100

    """
    return ATR(high, low, close, timeperiod) / close
