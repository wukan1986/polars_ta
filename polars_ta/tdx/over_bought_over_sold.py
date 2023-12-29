from polars import Expr

from polars_ta.ta.momentum import RSV
from polars_ta.ta.price import TYPPRICE
from polars_ta.tdx.arithmetic import ABS
from polars_ta.tdx.choice import IF
from polars_ta.tdx.reference import DIFF
from polars_ta.tdx.reference import MA
from polars_ta.tdx.reference import MAX
from polars_ta.tdx.reference import REF
from polars_ta.tdx.reference import SMA
from polars_ta.tdx.reference import SUM
from polars_ta.tdx.reference import TR
from polars_ta.tdx.statistic import AVEDEV


def ATR(HIGH: Expr, LOW: Expr, CLOSE: Expr, N: int = 14) -> Expr:
    """

    Notes
    -----
    与talib.ATR不同

    """
    return MA(TR(HIGH, LOW, CLOSE), N)


def BIAS(CLOSE: Expr, N: int = 6) -> Expr:
    """
    BIAS1 :(CLOSE-MA(CLOSE,N1))/MA(CLOSE,N1)*100;
    BIAS2 :(CLOSE-MA(CLOSE,N2))/MA(CLOSE,N2)*100;
    BIAS3 :(CLOSE-MA(CLOSE,N3))/MA(CLOSE,N3)*100;

    """
    return CLOSE / MA(CLOSE, N) - 1  # * 100


def CCI(HIGH: Expr, LOW: Expr, CLOSE: Expr, N: int = 14) -> Expr:
    """
    TYP:=(HIGH+LOW+CLOSE)/3;
    CCI:(TYP-MA(TYP,N))*1000/(15*AVEDEV(TYP,N));

    Notes
    -----
    AVEDEV计算慢，少用

    """
    TYP = TYPPRICE(HIGH, LOW, CLOSE)
    return (TYP - MA(TYP, N)) / (0.015 * AVEDEV(TYP, N))


def KDJ(HIGH: Expr, LOW: Expr, CLOSE: Expr, N: int = 9, M1: int = 3, M2: int = 3) -> Expr:
    """
    RSV:=(CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100;
    K:SMA(RSV,M1,1);
    D:SMA(K,M2,1);
    J:3*K-2*D;

    """
    rsv = RSV(HIGH, LOW, CLOSE, N)
    k = SMA(rsv, M1, 1)
    d = SMA(k, M2, 1)
    j = k * 3 - d * 2
    # return k, d, j
    return j


def MTM(CLOSE: Expr, N: int = 12) -> Expr:
    """
    MTM:CLOSE-REF(CLOSE,MIN(BARSCOUNT(C),N));
    MTMMA:MA(MTM,M);
    """
    # return CLOSE - REF(CLOSE, N)
    return DIFF(CLOSE, N)


def RSI(CLOSE: Expr, N: int = 6) -> Expr:
    """
    LC:=REF(CLOSE,1);
    RSI1:SMA(MAX(CLOSE-LC,0),N1,1)/SMA(ABS(CLOSE-LC),N1,1)*100;
    RSI2:SMA(MAX(CLOSE-LC,0),N2,1)/SMA(ABS(CLOSE-LC),N2,1)*100;
    RSI3:SMA(MAX(CLOSE-LC,0),N3,1)/SMA(ABS(CLOSE-LC),N3,1)*100;
    """
    LC = REF(CLOSE, 1)
    DIF = CLOSE - LC
    return SMA(MAX(DIF, 0), N, 1) / SMA(ABS(DIF), N, 1)  # * 100


def MFI(CLOSE: Expr, HIGH: Expr, LOW: Expr, VOL: Expr, N: int = 14) -> Expr:
    """

    TYP := (HIGH + LOW + CLOSE)/3;
    V1:=SUM(IF(TYP>REF(TYP,1),TYP*VOL,0),N)/SUM(IF(TYP<REF(TYP,1),TYP*VOL,0),N);
    MFI:100-(100/(1+V1));

    """
    TYP = TYPPRICE(HIGH, LOW, CLOSE)
    LT = REF(TYP, 1)
    V1 = SUM(IF(TYP > LT, TYP * VOL, 0), N) / SUM(IF(TYP < LT, TYP * VOL, 0), N)
    return (1 - (1 / (1 + V1)))  # * 100
