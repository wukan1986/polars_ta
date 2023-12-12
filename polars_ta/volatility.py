import polars as pl


def TRANGE(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """

    Notes
    -----
    第0位置为max(x,nan,nan)=x,比talib多一个值

    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pl.max_horizontal(tr1, tr2, tr3)


def ATR(high: pl.Expr, low: pl.Expr, close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    # TODO 需要认真修改
    return TRANGE(high, low, close).ewm_mean(span=timeperiod, adjust=False, min_periods=timeperiod)


def NATR(high: pl.Expr, low: pl.Expr, close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    """

    Notes
    -----
    talib.ATR版相当于多乘了100

    """
    return ATR(high, low, close, timeperiod) / close
