from polars import Expr, when

from polars_ta import TA_EPSILON
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
    Lower band:
    下轨:(N-LLVBARS(L,N))/N*100,COLORGREEN;
    """
    return 1 - ts_arg_min(low, timeperiod, reverse=True) / timeperiod


def AROON_aroonup(high: Expr, low: Expr, timeperiod: int = 14) -> Expr:
    """
    Upper band:
    上轨:(N-HHVBARS(H,N))/N*100,COLORRED;

    Notes
    -----
    You cannot use pd.Series.rolling().arg_max() with reverse order, which leads to a larger result when there are two or more high points
    so we don't use pandas
    pd.Series.rolling().arg_max()没有逆序，导致出现两个及以上最高点时，结果偏大

    """
    return 1 - ts_arg_max(high, timeperiod, reverse=True) / timeperiod


def MACD_macd(close: Expr, fastperiod: int = 12, slowperiod: int = 26) -> Expr:
    """MACD

    Notes
    -----
    When counting how many data we have
    we refer to `slowperiod`, while `talib.MACD` refers to `fastperiod`

    talib.MACD有效数据按fastperiod，而本项目按slowperiod

    """
    return EMA(close, fastperiod) - EMA(close, slowperiod)


def MACD_macdhist(close: Expr, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Expr:
    """MACD

    Notes
    -----
    Chinese version is multiplied by 2

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
    """MOM = (price - prevPrice) [Momentum]

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_MOM.c#L200

    """
    return ts_delta(close, timeperiod)


def PPO(close: Expr, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> Expr:
    if matype == 0:
        return SMA(close, fastperiod) / SMA(close, slowperiod) - 1
    else:
        return EMA(close, fastperiod) / EMA(close, slowperiod) - 1


def ROC(close: Expr, timeperiod: int = 10) -> Expr:
    """ROC = ((price/prevPrice)-1)*100 [Rate of change]

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_ROC.c#L200

    """
    return ROCP(close, timeperiod) * 100


def ROCP(close: Expr, timeperiod: int = 10) -> Expr:
    """ROCP = (price-prevPrice)/prevPrice [Rate of change Percentage]

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_ROCP.c#L202

    """
    return ts_returns(close, timeperiod)


def ROCR(close: Expr, timeperiod: int = 10) -> Expr:
    """ROCR = (price/prevPrice) [Rate of change ratio]

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_ROCR.c#L203

    """
    return close / close.shift(timeperiod)


def ROCR100(close: Expr, timeperiod: int = 10) -> Expr:
    """ROCR100 = (price/prevPrice)*100 [Rate of change ratio 100 Scale]

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_ROCR100.c#L203

    """
    return ROCR(close, timeperiod) * 100


def RSI(close: Expr, timeperiod: int = 14) -> Expr:
    dif = close.diff().fill_null(0)
    return RMA(max_(dif, 0), timeperiod) / (RMA(dif.abs(), timeperiod) + TA_EPSILON)  # * 100


def STOCHF_fastd(high: Expr, low: Expr, close: Expr, fastk_period: int = 5, fastd_period: int = 3) -> Expr:
    return SMA(RSV(high, low, close, fastk_period), fastd_period)


def TRIX(close: Expr, timeperiod: int = 30) -> Expr:
    EMA1 = EMA(close, timeperiod)
    EMA2 = EMA(EMA1, timeperiod)
    EMA3 = EMA(EMA2, timeperiod)
    return ROCP(EMA3, 1)


def RSV(high: Expr, low: Expr, close: Expr, timeperiod: int = 5) -> Expr:
    """RSV=STOCHF_FASTK

                      (Today's Close - LowestLow)
    FASTK(Kperiod) =  --------------------------- * 100
                       (HighestHigh - LowestLow)

    Notes
    -----
    RSV = STOCHF_FASTK。没乘以100

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_STOCHF.c#L279

    """
    a = MAX(high, timeperiod)
    b = MIN(low, timeperiod)

    # return (close - b) / (a - b + TA_EPSILON)
    return when(a != b).then((close - b) / (a - b)).otherwise(0)


def WILLR(high: Expr, low: Expr, close: Expr, timeperiod: int = 14) -> Expr:
    """威廉指标

    Notes
    -----
    WILLR=1-RSV

    References
    ----------
    - https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_WILLR.c#L294
    - https://www.investopedia.com/terms/w/williamsr.asp
    - https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r


    """
    a = MAX(high, timeperiod)
    b = MIN(low, timeperiod)

    # return (a - close) / (a - b + TA_EPSILON)
    return when(a != b).then((a - close) / (a - b)).otherwise(0)
