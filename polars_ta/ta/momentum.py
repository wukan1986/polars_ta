from polars import Expr

from polars_ta.ta.operators import MAX
from polars_ta.ta.operators import MIN
from polars_ta.ta.overlap import EMA
from polars_ta.ta.overlap import RMA
from polars_ta.ta.overlap import SMA
from polars_ta.wq.arithmetic import max_
from polars_ta.wq.time_series import ts_delta, ts_arg_max, ts_arg_min
from polars_ta.wq.time_series import ts_returns


def APO(close: Expr, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> Expr:
    if matype == 0:
        return SMA(close, fastperiod) - SMA(close, slowperiod)
    else:
        return EMA(close, fastperiod) - EMA(close, slowperiod)


def AROON_aroondown(high: Expr, low: Expr, timeperiod: int = 14) -> Expr:
    """
    下轨:(N-LLVBARS(L,N))/N*100,COLORGREEN;
    """
    return 1 - ts_arg_min(low, timeperiod, reverse=True) / timeperiod


def AROON_aroonup(high: Expr, low: Expr, timeperiod: int = 14) -> Expr:
    """
    上轨:(N-HHVBARS(H,N))/N*100,COLORRED;

    Notes
    -----
    arg_max没有逆序，导致出现两个及以上最高点时，结果偏大

    """
    return 1 - ts_arg_max(high, timeperiod, reverse=True) / timeperiod


def MACD_macd(close: Expr, fastperiod: int = 12, slowperiod: int = 26) -> Expr:
    """

    Notes
    -----
    talib.MACD有效数据按fastperiod，而本项目按slowperiod

    """
    return EMA(close, fastperiod) - EMA(close, slowperiod)


def MACD_macdhist(close: Expr, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Expr:
    """

    Notes
    -----
    中国版多了乘2

    """
    macd = MACD_macd(close, fastperiod, slowperiod)
    signal = EMA(macd, signalperiod)
    return macd - signal


def MACD_macdsignal(close: Expr, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Expr:
    macd = MACD_macd(close, fastperiod, slowperiod)
    signal = EMA(macd, signalperiod)
    return signal


def MOM(close: Expr, timeperiod: int = 10) -> Expr:
    return ts_delta(close, timeperiod)


def PPO(close: Expr, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> Expr:
    if matype == 0:
        return SMA(close, fastperiod) / SMA(close, slowperiod) - 1
    else:
        return EMA(close, fastperiod) / EMA(close, slowperiod) - 1


def ROC(close: Expr, timeperiod: int = 10) -> Expr:
    return ROCP(close, timeperiod) * 100


def ROCP(close: Expr, timeperiod: int = 10) -> Expr:
    return ts_returns(close, timeperiod)


def ROCR(close: Expr, timeperiod: int = 10) -> Expr:
    return close / close.shift(timeperiod)


def ROCR100(close: Expr, timeperiod: int = 10) -> Expr:
    return ROCR(close, timeperiod) * 100


def RSI(close: Expr, timeperiod: int = 14) -> Expr:
    dif = close.diff().fill_null(0)
    return RMA(max_(dif, 0), timeperiod) / RMA(dif.abs(), timeperiod)  # * 100


def RSV(high: Expr, low: Expr, close: Expr, timeperiod: int = 5) -> Expr:
    """

    Notes
    -----
    又名STOCHF_fastk, talib.STOCHF版相当于多乘了100，与WILLR指标又很像

    """
    a = MAX(high, timeperiod)
    b = MIN(low, timeperiod)

    return (close - b) / (a - b)


def STOCHF_fastd(high: Expr, low: Expr, close: Expr, fastk_period: int = 5, fastd_period: int = 3) -> Expr:
    return SMA(RSV(high, low, close, fastk_period), fastd_period)


def TRIX(close: Expr, timeperiod: int = 30) -> Expr:
    EMA1 = EMA(close, timeperiod)
    EMA2 = EMA(EMA1, timeperiod)
    EMA3 = EMA(EMA2, timeperiod)
    return ROCP(EMA3, 1)


def WILLR(high: Expr, low: Expr, close: Expr, timeperiod: int = 14) -> Expr:
    """

    Notes
    -----
    talib.WILLR版相当于多乘了-100，但个人认为没有必要

    References
    ----------
    https://www.investopedia.com/terms/w/williamsr.asp
    https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r

    """
    a = MAX(high, timeperiod)
    b = MIN(low, timeperiod)

    return (a - close) / (a - b)
