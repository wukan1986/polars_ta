from polars import Expr

from polars_ta.tdx.choice import IF
from polars_ta.tdx.reference import REF
from polars_ta.tdx.reference import SUM
from polars_ta.tdx.reference import CUMSUM


def OBV(CLOSE: Expr, VOL: Expr) -> Expr:
    """
    VA:=IF(CLOSE>REF(CLOSE,1),VOL,-VOL);
    OBV:SUM(IF(CLOSE=REF(CLOSE,1),0,VA),0);
    MAOBV:MA(OBV,M);
    """
    LC = REF(CLOSE, 1)
    VA = IF(CLOSE - LC, VOL, -VOL)
    return CUMSUM(IF(CLOSE == LC, 0, VA))


def VR(CLOSE: Expr, VOL: Expr, N: int = 26) -> Expr:
    """
    TH:=SUM(IF(CLOSE>REF(CLOSE,1),VOL,0),N);
    TL:=SUM(IF(CLOSE<REF(CLOSE,1),VOL,0),N);
    TQ:=SUM(IF(CLOSE=REF(CLOSE,1),VOL,0),N);
    VR:100*(TH*2+TQ)/(TL*2+TQ);
    MAVR:MA(VR,M);
    """
    LC = REF(CLOSE, 1)
    TH = SUM(IF(CLOSE > LC, VOL, 0), N)
    TL = SUM(IF(CLOSE < LC, VOL, 0), N)
    TQ = SUM(IF(CLOSE == LC, VOL, 0), N)

    return (TH * 2 + TQ) / (TL * 2 + TQ)  # * 100
