import polars as pl

from polars_ta.tdx.reference import MA
from polars_ta.tdx.reference import REF


def DPO(CLOSE: pl.Expr, N: int = 20) -> pl.Expr:
    """
    DPO:CLOSE-REF(MA(CLOSE,N),N/2+1);
    MADPO:MA(DPO,M);
    """

    return CLOSE - REF(MA(CLOSE, N), N // 2 + 1)


def EMV(HIGH: pl.Expr, LOW: pl.Expr, VOL: pl.Expr, N: int = 14) -> pl.Expr:
    """
    VOLUME:=MA(VOL,N)/VOL;
    MID:=100*(HIGH+LOW-REF(HIGH+LOW,1))/(HIGH+LOW);
    EMV:MA(MID*VOLUME*(HIGH-LOW)/MA(HIGH-LOW,N),N);
    MAEMV:MA(EMV,M);
    """
    ADD = HIGH + LOW
    SUB = HIGH - LOW

    VOLUME = MA(VOL, N) / VOL
    MID = 100 * (ADD - REF(ADD, 1)) / ADD
    return MA(MID * VOLUME * SUB / MA(SUB, N), N)
