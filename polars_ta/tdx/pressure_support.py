from polars import Expr

from polars_ta.tdx.arithmetic import SQRT
from polars_ta.tdx.reference import MA
from polars_ta.tdx.statistic import STDP


def BOLL(close: Expr, M: int = 20, N: int = 2) -> Expr:
    """
    BOLL:MA(CLOSE,M);
    UB:BOLL+2*STD(CLOSE,M);
    LB:BOLL-2*STD(CLOSE,M);
    """
    ma = MA(close, M)
    # it should be total standard deviation, the value is smaller than sample standard deviation.
    # 这里是总体标准差，值比样本标准差小。部分软件使用样本标准差是错误的，
    std = STDP(close, M)
    return ma + std * N


def BOLL_M(close: Expr, M: int = 20, N: int = 2) -> Expr:
    """
    {参数 N: 2  250  20 }
    MID:=MA(C,N);
    VART1:=POW((C-MID),2);
    VART2:=MA(VART1,N);
    VART3:=SQRT(VART2);
    UPPER:=MID+2*VART3;
    LOWER:=MID-2*VART3;
    BOLL:REF(MID,1),COLORFFFFFF;
    UB:REF(UPPER,1),COLOR00FFFF;
    LB:REF(LOWER,1),COLORFF00FF;
    """
    ma = MA(close, M)
    # 这里var不一样
    # close-close.mean()与close-MA(close)的区别
    std = SQRT(MA((close - ma) ** 2, M))
    return ma + std * N
