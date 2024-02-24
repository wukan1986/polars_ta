from polars import Expr

from polars_ta import TA_EPSILON
from polars_ta.tdx.arithmetic import ABS
from polars_ta.tdx.choice import IF
from polars_ta.tdx.reference import MA
from polars_ta.tdx.reference import REF
from polars_ta.tdx.reference import SUM
from polars_ta.tdx.reference import TR


def DPO(CLOSE: Expr, N: int = 20) -> Expr:
    """
    DPO:CLOSE-REF(MA(CLOSE,N),N/2+1);
    MADPO:MA(DPO,M);
    """

    return CLOSE - REF(MA(CLOSE, N), N // 2 + 1)


def EMV(HIGH: Expr, LOW: Expr, VOL: Expr, N: int = 14) -> Expr:
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
    return MA(MID * VOLUME * SUB / (MA(SUB, N) + TA_EPSILON), N)


def PLUS_DM(HIGH: Expr, LOW: Expr, N: int = 14) -> Expr:
    """
    MTR:=SUM(MAX(MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1))),ABS(REF(CLOSE,1)-LOW)),N);
    HD :=HIGH-REF(HIGH,1);
    LD :=REF(LOW,1)-LOW;
    DMP:=SUM(IF(HD>0&&HD>LD,HD,0),N);
    DMM:=SUM(IF(LD>0&&LD>HD,LD,0),N);
    """
    HD = HIGH - REF(HIGH, 1)
    LD = REF(LOW, 1) - LOW
    DMP = SUM(IF((HD > 0) & (HD > LD), HD, 0), N)
    return DMP


def MINUS_DM(HIGH: Expr, LOW: Expr, N: int = 14) -> Expr:
    HD = HIGH - REF(HIGH, 1)
    LD = REF(LOW, 1) - LOW
    DMM = SUM(IF((LD > 0) & (LD > HD), LD, 0), N)
    return DMM


def PLUS_DI(HIGH: Expr, LOW: Expr, CLOSE: Expr, N: int = 14) -> Expr:
    """
    MTR:=SUM(MAX(MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1))),ABS(REF(CLOSE,1)-LOW)),N);
    HD :=HIGH-REF(HIGH,1);
    LD :=REF(LOW,1)-LOW;
    DMP:=SUM(IF(HD>0&&HD>LD,HD,0),N);
    DMM:=SUM(IF(LD>0&&LD>HD,LD,0),N);
    PDI: DMP*100/MTR;
    MDI: DMM*100/MTR;
    """
    MTR = SUM(TR(HIGH, LOW, CLOSE), N)
    DMP = PLUS_DM(HIGH, LOW, N)
    PDI = DMP / MTR
    return PDI  # * 100


def MINUS_DI(HIGH: Expr, LOW: Expr, CLOSE: Expr, N: int = 14) -> Expr:
    MTR = SUM(TR(HIGH, LOW, CLOSE), N)
    DMM = MINUS_DM(HIGH, LOW, N)
    MDI = DMM / MTR
    return MDI  # * 100


def ADX(HIGH: Expr, LOW: Expr, CLOSE: Expr, N: int = 14, M: int = 6) -> Expr:
    """
    MTR:=SUM(MAX(MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1))),ABS(REF(CLOSE,1)-LOW)),N);
    HD :=HIGH-REF(HIGH,1);
    LD :=REF(LOW,1)-LOW;
    DMP:=SUM(IF(HD>0&&HD>LD,HD,0),N);
    DMM:=SUM(IF(LD>0&&LD>HD,LD,0),N);
    PDI: DMP*100/MTR;
    MDI: DMM*100/MTR;
    ADX: MA(ABS(MDI-PDI)/(MDI+PDI)*100,M);
    ADXR:(ADX+REF(ADX,M))/2;
    """
    MTR = SUM(TR(HIGH, LOW, CLOSE), N)
    HD = HIGH - REF(HIGH, 1)
    LD = REF(LOW, 1) - LOW
    DMP = SUM(IF((HD > 0) & (HD > LD), HD, 0), N)
    DMM = SUM(IF((LD > 0) & (LD > HD), LD, 0), N)
    PDI = DMP / MTR  # * 100
    MDI = DMM / MTR  # * 100
    return MA(ABS(MDI - PDI) / (MDI + PDI), M)  # * 100


def ADXR(HIGH: Expr, LOW: Expr, CLOSE: Expr, N: int = 14, M: int = 6) -> Expr:
    adx = ADX(HIGH, LOW, CLOSE, N, M)
    adxr = (adx + REF(adx, M)) / 2
    return adxr
