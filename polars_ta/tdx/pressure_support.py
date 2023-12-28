from polars import Expr

from polars_ta.tdx.reference import MA
from polars_ta.tdx.statistic import STDP


def BOLL(close: Expr, M: int = 20, N: int = 2) -> Expr:
    """
    BOLL:MA(CLOSE,M);
    UB:BOLL+2*STD(CLOSE,M);
    LB:BOLL-2*STD(CLOSE,M);
    """
    ma = MA(close, M)
    # 这里是总体标准差，值比样本标准差小。部分软件使用样本标准差是错误的，
    _std = STDP(close, M)
    UB = ma + _std * N
    return UB
