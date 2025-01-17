from polars import Expr

from polars_ta.tdx.arithmetic import ABS, BETWEEN
from polars_ta.tdx.arithmetic import MAX
from polars_ta.tdx.arithmetic import MIN
from polars_ta.tdx.choice import IF
from polars_ta.tdx.logical import EVERY, CROSS, LAST, EXIST, NOT
from polars_ta.tdx.reference import COUNT, BARSLASTCOUNT, BARSLAST, EMA, FILTER, LOWRANGE
from polars_ta.tdx.reference import HHV
from polars_ta.tdx.reference import LLV
from polars_ta.tdx.reference import MA
from polars_ta.tdx.reference import REF
from polars_ta.wq.time_series import ts_cum_max, ts_cum_min


def 早晨之星(OPEN: Expr, CLOSE: Expr) -> Expr:
    """MSTAR 早晨之星"""
    A1 = REF(CLOSE, 2) / REF(OPEN, 2) < 0.95
    A2 = REF(OPEN, 1) < REF(CLOSE, 2)
    A3 = ABS(REF(OPEN, 1) - REF(CLOSE, 1)) / REF(CLOSE, 1) < 0.03
    A4 = CLOSE / OPEN > 1.05
    A5 = CLOSE > REF(CLOSE, 2)
    return A1 & A2 & A3 & A4 & A5


def 剑(OPEN: Expr, HIGH: Expr, LOW: Expr, CLOSE: Expr, VOL: Expr, CAPITAL: Expr) -> Expr:
    """SWORD 剑"""
    AA = (VOL > REF(VOL, 1)) | (VOL > CAPITAL)
    BB = (OPEN >= REF(HIGH, 1)) & (REF(HIGH, 1) > REF(HIGH, 2) * 1.06)
    CC = CLOSE > (REF(CLOSE, 1) - REF(CLOSE, 1) * 0.01)
    DD = (CLOSE < HIGH * 0.965) & (HIGH > OPEN * 1.05)
    EE = (LOW < OPEN) & (LOW < CLOSE) & (HIGH > REF(CLOSE, 1) * 1.06)
    FF = (HIGH - MAX(OPEN, CLOSE)) / 2 > (MIN(OPEN, CLOSE) - LOW)
    GG = (ABS(OPEN - CLOSE)) / 2 < (MIN(OPEN, CLOSE) - LOW)
    return AA & BB & CC & DD & EE & FF & GG


def 天量法则(OPEN: Expr, CLOSE: Expr) -> Expr:
    """TLFZ 天量法则"""
    A1 = CLOSE > OPEN
    A2 = HHV(CLOSE, 50) == CLOSE
    # DYNAINFO(37) > 0.1 & &
    # DYNAINFO(13) < 0.14;
    raise


def 四串阴(OPEN: Expr, CLOSE: Expr) -> Expr:
    """GREEN4 四串阴"""
    return EVERY(CLOSE < OPEN, 4)


def 四串阳(OPEN: Expr, CLOSE: Expr) -> Expr:
    """RED4 四串阳"""
    return EVERY(CLOSE > OPEN, 4)


def 鸳鸯底(O: Expr, LOW: Expr, C: Expr, V: Expr, N: int = 50) -> Expr:
    """YYD 鸳鸯底"""
    return (C > O) & (REF(C, 1) < REF(O, 1)) & (C > REF(O, 1)) & (V > REF(V, 1)) & EXIST(LOWRANGE(LOW) > N, 3)


def 出水芙蓉(OPEN: Expr, CLOSE: Expr, S: int = 20, M: int = 40, N: int = 60) -> Expr:
    """CSFR 出水芙蓉"""
    AAA = CLOSE > OPEN
    BBB = AAA & (CLOSE > MA(CLOSE, S)) & (CLOSE > MA(CLOSE, M)) & (CLOSE > MA(CLOSE, N))
    CCC = BBB & (OPEN < MA(CLOSE, M)) & (OPEN < MA(CLOSE, N))
    return CCC & (CLOSE - OPEN > 0.0618 * CLOSE)


def 出水芙蓉II(C: Expr, V: Expr, N: float = 0.05, M: float = 2.0) -> Expr:
    """CSFR2 出水芙蓉II"""
    ZF = C / REF(C, 1)
    FLTJ = V > REF(V, 1) * M
    A1 = CROSS(C, MA(C, 5)) & CROSS(C, MA(C, 10)) & CROSS(C, MA(C, 20)) & CROSS(C, MA(C, 60))
    return (ZF > 1 + N) & FLTJ & A1


def 近日创历史新高(HIGH: Expr, N: int = 3, M: int = 0) -> Expr:
    """NHIGH 近日创历史新高"""
    if M == 0:
        return HHV(HIGH, N) == ts_cum_max(HIGH)
    else:
        return HHV(HIGH, N) == HHV(HIGH, M)


def 近日创历史新低(LOW: Expr, N: int = 3, M: int = 0) -> Expr:
    """NLOW 近日创历史新低"""
    if M == 0:
        return LLV(LOW, N) == ts_cum_min(LOW)
    else:
        return LLV(LOW, N) == LLV(LOW, M)


def 旭日初升(CLOSE: Expr, VOL: Expr, N: int = 120) -> Expr:
    """XRDS 旭日初升"""
    BUY1 = LAST(CLOSE < MA(CLOSE, N), 0, 5)
    return (CLOSE > MA(CLOSE, N)) & (VOL > MA(VOL, 5) * 2) & BUY1


def 蜻蜓点水(CLOSE: Expr, N: int = 120) -> Expr:
    """QTDS 蜻蜓点水"""
    BUY1 = LAST(CLOSE > MA(CLOSE, N), 0, 5)
    BUY2 = EXIST(CLOSE < MA(CLOSE, N), 5)
    return (CLOSE > MA(CLOSE, N)) & BUY1 & BUY2


def 均线多头排列(OPEN: Expr, CLOSE: Expr, N: int = 5, N1: int = 10, N2: int = 20, N3: int = 30) -> Expr:
    """DTPL 均线多头排列"""
    A1 = MA(CLOSE, N)
    A2 = MA(CLOSE, N1)
    A3 = MA(CLOSE, N2)
    A4 = MA(CLOSE, N3)
    return (CLOSE > A1) & (A1 > A2) & (A2 > A3) & (A3 > A4) & (CLOSE > OPEN)


def 均线空头排列(OPEN: Expr, CLOSE: Expr, N: int = 5, N1: int = 10, N2: int = 20, N3: int = 30) -> Expr:
    """KTPL 均线空头排列"""
    A1 = MA(CLOSE, N)
    A2 = MA(CLOSE, N1)
    A3 = MA(CLOSE, N2)
    A4 = MA(CLOSE, N3)
    return (CLOSE < A1) & (A1 < A2) & (A2 < A3) & (A3 < A4) & (CLOSE < OPEN)


def 强势整理(OPEN: Expr, CLOSE: Expr, N: int = 2, M: float = 0.05) -> Expr:
    """QSZL 强势整理"""
    A1 = ABS(CLOSE - OPEN) / OPEN < 0.015
    A2 = COUNT(A1, N) == N
    A3 = (REF(OPEN, N) < REF(CLOSE, N)) & (REF(CLOSE, N) / REF(CLOSE, N + 1) > 1 + M)
    return A2 & A3


def 高开大阴线(OPEN: Expr, CLOSE: Expr, N: float = 0.06, M: float = 0.04) -> Expr:
    """W103 高开大阴线"""
    A1 = OPEN / REF(CLOSE, 1) >= 1 + M
    A2 = CLOSE / OPEN <= 1 - N
    return A1 & A2


def 低开大阳线(OPEN: Expr, CLOSE: Expr, N: float = 0.06, M: float = 0.04) -> Expr:
    """W104 低开大阳线"""
    A1 = OPEN / REF(CLOSE, 1) <= 1 - M
    A2 = CLOSE / OPEN >= 1 + N
    return A1 & A2


def 跳空缺口选股(HIGH: Expr, LOW: Expr) -> Expr:
    """W105 跳空缺口选股"""
    A1 = HIGH < (REF(LOW, 1) - 0.001)
    A2 = LOW > (REF(HIGH, 1) + 0.001)
    return A1 | A2


def 单阳不破选股(O: Expr, H: Expr, L: Expr, C: Expr, N1: int = 2, N2: int = 7) -> Expr:
    """W106 单阳不破选股
    """
    A0 = ((C > O * 1.08) | (C > REF(C, 1) * 1.08)) & NOT(H == L) & NOT(H == C & H == O)  # {大阳超8%，排除当天一字、T字板}
    A1 = A0 & BARSLASTCOUNT(A0) == 1
    A2 = BARSLAST(A1)  # {距离大阳几根K}
    ZCX = REF(O, A2)  # {获取大阳位置的开盘价作为支撑线}
    ZHX = REF(C, A2)  # {获取大阳位置的收盘价作为选股最高区间}
    ZD = LLV(L, A2)  # {大阳之后的最低价}
    ZH = HHV(H, A2)  # {大阳之后的最高价}
    A3 = BARSLASTCOUNT(ZD >= ZCX)
    return (A3 <= N2) & (A3 > N1) & BETWEEN(C, ZCX, ZHX) & (ZH < ZHX) & (A2 > 0)


def 回补跳空向上缺口(O: Expr, H: Expr, L: Expr, C: Expr, N1: int = 2, N2: int = 7) -> Expr:
    """W107 回补跳空向上缺口"""
    raise


def 揉搓线(O: Expr, H: Expr, L: Expr, C: Expr, V: Expr, N: int = 50) -> Expr:
    """RUBLINE 揉搓线
    """
    A1 = (REF(H, 1) - MAX(REF(C, 1), REF(O, 1))) / (REF(H, 1) - REF(L, 1)) * 100 > N  # {上影线占K线的N % 以上}
    A2 = (MIN(O, C) - L) / (H - L) * 100 > N  # {下影线占K线的N % 以上}
    A3 = ABS(C - REF(C, 1)) / MIN(C, REF(C, 1)) * 100 < 2  # {下影K的跌幅不能超过2 %}
    A4 = REF(C, 2) > REF(C, 3)  # {揉搓形态前收涨}
    A5 = V < REF(V, 1)  # {缩量}
    return A1 & A2 & A3 & A4 & A5


def 老鸭头(L: Expr, C: Expr, V: Expr) -> Expr:
    """OLDDUCK 老鸭头
    """
    E1 = EMA(C, 13)
    E2 = EMA(C, 55)
    A1 = (COUNT(E1 < REF(E1, 1), 5) >= 3) & (E1 > REF(E1, 1))
    A2 = (COUNT(E2 > REF(E2, 1), 13) >= 8) & (E2 < REF(E2, 1))
    A3 = LLV((L / E2 - 1), 13) <= 0.1
    A4 = COUNT(E1 > E2, 13) == 13
    A5 = COUNT(C > E2, 5) == 5
    A6 = CROSS(C, E1)
    A7 = V > MA(V, 5)
    YT = A1 & A2 & A3 & A4 & A5 & A6 & A7
    LYT = FILTER(YT, 10)
    # FXG:=FINANCE(42)>100;
    # NTP:=DYNAINFO(8)>0;
    # LYT AND FXG AND NTP;
    return LYT


def 仙人指路(O: Expr, H: Expr, C: Expr) -> Expr:
    """WISEWAY 仙人指路
    """
    TUPO = 0.5 * (C + H) > HHV(REF(C, 1), 60)
    SSQS = (MA(C, 5) > MA(C, 60)) & (MA(C, 10) > MA(C, 60))
    YINX = ((H - MAX(O, C)) / REF(C, 1) > 0.045) & (ABS(C - O) / REF(C, 1) < 0.035) & ((H - MAX(O, C)) > 2.0 * ABS(C - O))
    QIANG1 = (REF(C, 1) / REF(C, 6) > 1.04) & (REF(C, 1) / REF(C, 6) < 1.18)
    QIANG2 = (REF(C, 1) / REF(C, 6) > 1.04) & (REF(C, 1) / REF(C, 6) < 1.27)
    QIANG = IF(True, QIANG2, QIANG1)  # TODO 要改
    QIANK = (REF((H - C), 1) / REF(C, 2) < 1.045) & (REF(C, 1) / REF(C, 2) > 0.97)
    return TUPO & SSQS & YINX & QIANG & QIANK


def 低点搜寻(HIGH: Expr, LOW: Expr, CLOSE: Expr, N: int = 5) -> Expr:
    """SP 低点搜寻"""
    W = MA((LLV(LOW, 45) - CLOSE) / (HHV(HIGH, 45) - LLV(LOW, 45)), N)
    return CROSS(-0.05, W)


def 突破(C: Expr, N1: int = 5, N2: int = 10, N3: int = 30) -> Expr:
    """TP 突破"""
    M1 = MA(C, N1)  # {短期参数：5}
    M2 = MA(C, N2)  # {中期参数：10}
    M3 = MA(C, N3)  # {长期参数：30}
    # {以下计算交叉点距今的天数}
    D1 = BARSLAST(CROSS(M1, M2))  # {短上穿中}
    D2 = BARSLAST(CROSS(M1, M3))  # {短上穿长}
    D3 = BARSLAST(CROSS(M2, M3))  # {中上穿长}
    T1 = CROSS(M2, M3)  # {今天中线上穿长线}
    T2 = (D1 >= D2) & (D2 >= D3)  # {交叉按指定的先后出现}
    T3 = COUNT(CROSS(M2, M1) | CROSS(M3, M2) | CROSS(M3, M1), D1) == 0  # {中间无夹杂其它交叉}
    T4 = REF(M1 < M3 & M2 < M3, D1 + 1)  # {短上穿中前一天短、中线在长线之下}
    return T1 & T2 & T3 & T4  # {价托确定};
