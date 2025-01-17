from polars import Expr

from polars_ta.tdx.arithmetic import ABS
from polars_ta.tdx.logical import EXIST, EVERY
from polars_ta.tdx.pattern import ts_WINNER_COST
from polars_ta.tdx.reference import COUNT
from polars_ta.tdx.reference import HHV
from polars_ta.tdx.reference import LLV
from polars_ta.tdx.reference import MA
from polars_ta.tdx.reference import REF
from polars_ta.tdx.reference import SUM
from polars_ta.wq.time_series import ts_cum_max, ts_cum_min


def N天内有跳空向上缺口(H: Expr, L: Expr, N: int = 1, M: float = 0.01) -> Expr:
    """C100 N天内有跳空向上缺口

    Parameters
    ----------
    N
        天数
    M
        涨幅

    """
    XSTK = L > REF(H, 1) * (100 + M)
    return EXIST(XSTK, N)


def N日内创新高(HIGH: Expr, N: int = 10) -> Expr:
    """C101 N日内创新高
    """
    return HHV(HIGH, N) == ts_cum_max(HIGH)


def N日内创新低(LOW: Expr, N: int = 10) -> Expr:
    """C102 N日内创新低
    """
    return LLV(LOW, N) == ts_cum_min(LOW)


def N日内阴线多于阳线(OPEN: Expr, CLOSE: Expr, N: int = 30, M: float = 0.6) -> Expr:
    """C103 M日内阴线多于阳线

    Parameters
    ----------
    N
        天数
    M
        比例

    """
    return COUNT(OPEN > CLOSE, N) / N >= M


def N日内阳线多于阴线(OPEN: Expr, CLOSE: Expr, N: int = 30, M: float = 0.6) -> Expr:
    """C104 M日内阳线多于阴线

    Parameters
    ----------
    N
        天数
    M
        比例

    """
    return COUNT(OPEN < CLOSE, N) / N >= M


def N日内上涨多于下跌(CLOSE: Expr, N: int = 120, M: float = 0.6) -> Expr:
    """C105 N日内上涨多于下跌

    Parameters
    ----------
    N
        天数
    M
        比例

    """
    return COUNT(CLOSE > REF(CLOSE, 1), N) / N >= M


def N日内下跌多于上涨(CLOSE: Expr, N: int = 120, M: float = 0.6) -> Expr:
    """C106 N日内下跌多于上涨

    Parameters
    ----------
    N
        天数
    M
        比例

    """
    return COUNT(CLOSE < REF(CLOSE, 1), N) / N >= M


def 连续N天收阳线(OPEN: Expr, CLOSE: Expr, N: int = 7) -> Expr:
    """C107 连续N天收阳线

    Parameters
    ----------
    N
        天数

    """
    return EVERY(CLOSE > OPEN, N)


def 连续N天收阴线(OPEN: Expr, CLOSE: Expr, N: int = 7) -> Expr:
    """C108 连续N天收阴线

    Parameters
    ----------
    N
        天数

    """
    return EVERY(OPEN > CLOSE, N)


def 单日放量(VOL: Expr, CAPITAL: Expr, N: float = 2, M: float = 0.15) -> Expr:
    """C110 单日放量

    Parameters
    ----------
    N
        5日平均成交量的倍数
    M
        流通股本的倍数

    """
    A1 = MA(VOL, 5)
    A2 = REF(A1, 1)
    C1 = VOL / A2 > N
    C2 = VOL / CAPITAL > M
    return C1 & C2


def 阶段缩量(VOL: Expr, CAPITAL: Expr, N: int = 20, M: float = 0.02) -> Expr:
    """C111 阶段缩量

    Parameters
    ----------
    VOL
        成交量
    CAPITAL
        流通股本

    Notes
    -----
    成交量与流通股本单位要一致，都为手，或者都为股

    """
    return SUM(VOL, N) / CAPITAL <= M


def 阶段放量(VOL: Expr, CAPITAL: Expr, N: int = 10, M: float = 2.0) -> Expr:
    """C112 阶段放量

    Parameters
    ----------
    VOL
        成交量
    CAPITAL
        流通股本

    """
    return SUM(VOL, N) / CAPITAL >= M


def 持续放量(VOL: Expr, M: int = 5) -> Expr:
    """C113 持续放量

    Parameters
    ----------
    VOL
        成交量

    """
    return EVERY(VOL >= REF(VOL, 1), M)


def 持续缩量(VOL: Expr, M: int = 5) -> Expr:
    """C114 持续缩量

    Parameters
    ----------
    VOL
        成交量

    """
    return COUNT(VOL <= REF(VOL, 1), M) == M


def 间隔放量(VOL: Expr, N: int = 30, N1: float = 4.0, N2: float = 2.0, N3: int = 3) -> Expr:
    """C115 间隔放量

    Parameters
    ----------
    VOL
        成交量
    N
        均量周期
    N1
        最小均量与最大均量的倍数
    N2
        成交量与均量的倍数
    N3
        满足N2时的次数

    """
    A = MA(VOL, 5)
    C1 = HHV(A, N) < N1 * LLV(A, N)  # 成交量最大与最小在一定范围内
    C2 = COUNT(VOL > N2 * A, N) > N3  # 成交量大于均量一定倍数
    return C1 & C2


def 放量上攻(CLOSE: Expr, VOL: Expr, CAPITAL: Expr,
             N: float = 0.01, N1: int = 3, N2: float = 0.2, N3: int = 4) -> Expr:
    """C116 放量上攻

    Parameters
    ----------
    CLOSE
        复权收盘价
    VOL
        成交量
    CAPITAL
        流通股本
    N
        涨幅
    N1, N2
        N1天内成交量大于N2*流通股本
    N3
        连续N3天满足涨幅

    """
    A = (CLOSE - REF(CLOSE, 1)) / REF(CLOSE, 1) >= N
    C1 = SUM(VOL, N1) / CAPITAL >= N2
    C2 = COUNT(VOL > REF(VOL, 1), N3) == N3
    C3 = COUNT(A, N3) == N3
    return C1 & C2 & C3


def 温和放量上攻(CLOSE: Expr, VOL: Expr, CAPITAL: Expr, N: int = 5) -> Expr:
    """C117 温和放量上攻

    Parameters
    ----------
    CLOSE
        复权收盘价
    VOL
        成交量
    CAPITAL
        流通股本
    N
        观察天数

    """
    A1 = CLOSE / REF(CLOSE, 1)
    A2 = (A1 > 1) & (A1 < 1.03)  # {股价小幅上扬}
    B1 = VOL / REF(VOL, 1)
    B2 = (B1 > 1) & (A1 < 2)  # {成交量小幅上扬}
    C1 = MA(VOL, N) / CAPITAL < 0.05  # 日成交量小于流通股本的5%
    C2 = COUNT(A2 & B2, N) / N > 0.6
    return C1 & C2


def 突然放量(VOL: Expr, N: int = 10, M: float = 3.0) -> Expr:
    """C118 突然放量

    Parameters
    ----------
    VOL
        成交量
    N
        观察天数
    M
        倍数

    """
    return VOL > REF(HHV(VOL, N), 1) * M


def 平台整理(CLOSE: Expr, N: int = 30, N1: float = 0.05) -> Expr:
    """C119 平台整理

    Parameters
    ----------
    N
        天数
    N1
        幅度

    """
    return HHV(CLOSE, N) / LLV(CLOSE, N) <= 1 + N1


def 小步碎阳(O: Expr, H: Expr, L: Expr, C: Expr, avg: Expr, turnover_ratio: Expr, N: int = 4) -> Expr:
    """C120 小步碎阳

    Parameters
    ----------
    avg
        成交均价
    turnover_ratio
        换手率
    N
        观察天数

    """
    AA = (C > REF(C, 1)) & (C > O)
    A1 = COUNT(AA, N) == N
    A2 = C / REF(C, N) < 1.05
    A3 = ts_WINNER_COST(H, L, avg, turnover_ratio, C, 0.5).struct[0] > 0.75

    return A1 & A2 & A3


def 突破长期盘整(HIGH: Expr, LOW: Expr, CLOSE: Expr, N: int = 30, N1: int = 5) -> Expr:
    """C123 突破长期盘整

    Parameters
    ----------
    N
        天数
    N1
        涨幅

    """
    C1 = REF(HHV(HIGH, N) / LLV(LOW, N), 1) <= N1 + 1
    C2 = CLOSE >= REF(HHV(HIGH, N), 1)

    return C1 & C2


def N天内出现以涨停收盘(收盘涨停: Expr, N: int = 10) -> Expr:
    """C128 N天内出现以涨停收盘

    Parameters
    ----------
    N
        天数

    """
    return EXIST(收盘涨停, N)


def N天内出现涨停(最高涨停: Expr, N: int = 20) -> Expr:
    """C129 N天内出现涨停

    Parameters
    ----------
    N
        天数

    """
    return EXIST(最高涨停, N)


def N天内经常涨停(收盘涨停: Expr, N: int = 100, M: int = 8) -> Expr:
    """C130 N天内经常涨停

    Parameters
    ----------
    N
        天数
    M
        涨停天数

    """
    return COUNT(收盘涨停, N) >= M


def 下跌多日再放量上涨(HIGH: Expr, CLOSE: Expr, VOL: Expr) -> Expr:
    """C131 下跌多日再放量上涨

    """
    A1 = REF(CLOSE, 5) > REF(CLOSE, 4)
    A2 = REF(CLOSE, 4) > REF(CLOSE, 3)
    A3 = REF(CLOSE, 3) > REF(CLOSE, 2)
    A4 = REF(CLOSE, 2) > REF(CLOSE, 1)
    A5 = (CLOSE > REF(HIGH, 1)) & (VOL > REF(VOL, 1))
    return A1 & A2 & A3 & A4 & A5


def 跳空高开或低开(O: Expr, H: Expr, L: Expr, C: Expr, N: float = 0.03) -> Expr:
    """C132 跳空高开或低开

    Parameters
    ----------
    N
        涨幅

    """
    if N > 0:
        A = (O > REF(H, 1)) & (O / REF(C, 1) > (1 + N))
        return A
    else:
        B = (O < REF(L, 1)) & (O / REF(C, 1) < (1 + N))
        return B


def 拉升后多日调整(C: Expr, N: int = 3, ZF: float = 0.09) -> Expr:
    """C133 拉升后多日调整

    Parameters
    ----------
    N
        天数
    ZF
        涨幅

    """
    C1 = REF(C, N) / REF(C, N + 1) > 1 + ZF
    C2 = EVERY(C < REF(C, 1), N)
    return C1 & C2


def 昨日底部十字星(O: Expr, H: Expr, L: Expr, C: Expr, N: int = 60) -> Expr:
    """C134 昨日底部十字星

    Parameters
    ----------
    N
        天数

    """

    C1 = L <= LLV(L, N)
    C2 = ABS(C - O) / (H - L) < 0.05
    C3 = H > L
    return REF(C1 & C2 & C3, 1)


def 价量渐低后阳包阴(O: Expr, C: Expr, V: Expr) -> Expr:
    """C135 价量渐低后阳包阴
    """
    A1 = REF(C, 4) > REF(C, 3)
    A2 = REF(C, 3) > REF(C, 2)
    A3 = REF(C, 2) > REF(C, 1)
    B1 = REF(V, 3) > REF(V, 2)
    B2 = REF(V, 2) > REF(V, 1)
    AA1 = A1 & A2 & A3 & B1 & B2
    AA2 = (REF(O, 1) > REF(C, 1)) & (C > REF(O, 1))
    AA3 = (C > O) & (V < REF(V, 1))

    return AA1 & AA2 & AA3
