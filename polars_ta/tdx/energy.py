from polars import Expr

from polars_ta import TA_EPSILON
from polars_ta.ta.price import MEDPRICE
from polars_ta.tdx.reference import MA
from polars_ta.tdx.reference import MAX
from polars_ta.tdx.reference import REF
from polars_ta.tdx.reference import SUM


def BRAR_AR(OPEN: Expr, HIGH: Expr, LOW: Expr, CLOSE: Expr, N: int = 26) -> Expr:
    """
    BR:SUM(MAX(0,HIGH-REF(CLOSE,1)),N)/SUM(MAX(0,REF(CLOSE,1)-LOW),N)*100;
    AR:SUM(HIGH-OPEN,N)/SUM(OPEN-LOW,N)*100;

    """

    AR = SUM(HIGH - OPEN, N) / (SUM(OPEN - LOW, N) + TA_EPSILON)  # * 100
    return AR


def BRAR_BR(OPEN: Expr, HIGH: Expr, LOW: Expr, CLOSE: Expr, N: int = 26) -> Expr:
    """
    BR:SUM(MAX(0,HIGH-REF(CLOSE,1)),N)/SUM(MAX(0,REF(CLOSE,1)-LOW),N)*100;
    AR:SUM(HIGH-OPEN,N)/SUM(OPEN-LOW,N)*100;

    """
    LC = REF(CLOSE, 1)
    BR = SUM(MAX(0, HIGH - LC), N) / (SUM(MAX(0, LC - LOW), N) + TA_EPSILON)  # * 100
    return BR


def CR(HIGH: Expr, LOW: Expr, N: int = 26) -> Expr:
    """
    MID:=REF(HIGH+LOW,1)/2;
    CR:SUM(MAX(0,HIGH-MID),N)/SUM(MAX(0,MID-LOW),N)*100;
    MA1:REF(MA(CR,M1),M1/2.5+1);
    MA2:REF(MA(CR,M2),M2/2.5+1);
    MA3:REF(MA(CR,M3),M3/2.5+1);
    MA4:REF(MA(CR,M4),M4/2.5+1);

    """
    MID = REF(MEDPRICE(HIGH, LOW), 1)
    return SUM(MAX(0, HIGH - MID), N) / (SUM(MAX(0, MID - LOW), N) + TA_EPSILON)  # *100


def PSY(CLOSE: Expr, N: int = 12) -> Expr:
    """
    PSY:COUNT(CLOSE>REF(CLOSE,1),N)/N*100;
    PSYMA:MA(PSY,M);

    """
    return MA(CLOSE > REF(CLOSE, 1), N)


def MASS(HIGH: Expr, LOW: Expr, N1: int = 9, N2: int = 25) -> Expr:
    """
    MASS:SUM(MA(HIGH-LOW,N1)/MA(MA(HIGH-LOW,N1),N1),N2);
    MAMASS:MA(MASS,M);

    """
    MHL = MA(HIGH - LOW, N1)
    return SUM(MHL / MA(MHL, N1), N2)
