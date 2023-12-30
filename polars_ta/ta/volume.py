from polars import Expr

from polars_ta.ta.overlap import EMA


def AD(high: Expr, low: Expr, close: Expr, volume: Expr) -> Expr:
    ad = ((close - low) - (high - close)) / (high - low) * volume
    return ad.cum_sum()


def ADOSC(high: Expr, low: Expr, close: Expr, volume: Expr, fastperiod: int = 3, slowperiod: int = 10) -> Expr:
    ad = AD(high, low, close, volume)
    return EMA(ad, fastperiod) - EMA(ad, slowperiod)


def OBV(close: Expr, volume: Expr) -> Expr:
    """"""
    # 第一个值用volume就与talib.OBV完全一样了
    obv = close.diff().sign().fill_null(1) * volume
    return obv.cum_sum()
