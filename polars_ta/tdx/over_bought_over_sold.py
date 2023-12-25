import polars as pl

from polars_ta.ta.momentum import WILLR as WR  # noqa
from polars_ta.ta.price import MEDPRICE  # noqa
from polars_ta.ta.price import TYPPRICE  # noqa
from polars_ta.tdx.reference import MA  # noqa
from polars_ta.tdx.reference import TR  # noqa


def ATR(high: pl.Expr, low: pl.Expr, close: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    return MA(TR(high, low, close), timeperiod)


def BIAS(close: pl.Expr, timeperiod: int) -> pl.Expr:
    """BIAS乖离率

    Parameters
    ----------
    x
    d:int
        常用参数：6,12,24

    Notes
    -----
    防止再做标准化，与TDX原版相比，少乘100

    """
    return close / MA(close, timeperiod) - 1  # * 100
