from polars import Expr

from polars_ta.tdx.pattern import ts_WINNER_COST
from polars_ta.wq.arithmetic import abs_
from polars_ta.wq.time_series import ts_sum, ts_delay, ts_count, ts_mean, ts_max, ts_min, ts_returns, ts_cum_max, ts_cum_min


def N天内有跳空向上缺口(high: Expr, low: Expr, N: int = 1, M: float = 0.01) -> Expr:
    """C100 N天内有跳空向上缺口

    Parameters
    ----------
    N
        天数
    M
        涨幅

    """
    C1 = low > ts_delay(high, 1) * (1 + M)
    return ts_count(C1, N) > 0


def N日内创新高(high: Expr, N: int = 10) -> Expr:
    """C101 N日内创新高
    """
    return ts_max(high, N) == ts_cum_max(high)


def N日内创新低(low: Expr, N: int = 10) -> Expr:
    """C102 N日内创新低
    """
    return ts_min(low, N) == ts_cum_min(low)


def N日内阴线多于阳线(open_: Expr, close: Expr, N: int = 30, M: float = 0.6) -> Expr:
    """C103 M日内阴线多于阳线

    Parameters
    ----------
    N
        天数
    M
        比例

    """
    return ts_mean(open_ > close, N) >= M


def N日内阳线多于阴线(open_: Expr, close: Expr, N: int = 30, M: float = 0.6) -> Expr:
    """C104 M日内阳线多于阴线

    Parameters
    ----------
    N
        天数
    M
        比例

    """
    return ts_mean(open_ < close, N) >= M


def N日内上涨多于下跌(close: Expr, N: int = 120, M: float = 0.6) -> Expr:
    """C105 N日内上涨多于下跌

    Parameters
    ----------
    N
        天数
    M
        比例

    """
    return ts_mean(close > ts_delay(close, 1), N) >= M


def N日内下跌多于上涨(close: Expr, N: int = 120, M: float = 0.6) -> Expr:
    """C106 N日内下跌多于上涨

    Parameters
    ----------
    N
        天数
    M
        比例

    """
    return ts_mean(close < ts_delay(close, 1), N) >= M


def 连续N天收阳线(open_: Expr, close: Expr, N: int = 7) -> Expr:
    """C107 连续N天收阳线

    Parameters
    ----------
    N
        天数

    """
    return ts_count(close > open_, N) == N


def 连续N天收阴线(open_: Expr, close: Expr, N: int = 7) -> Expr:
    """C108 连续N天收阴线

    Parameters
    ----------
    N
        天数

    """
    return ts_count(open_ > close, N) == N


def 单日放量(volume: Expr, capital: Expr, N: float = 2, M: float = 0.15) -> Expr:
    """C110 单日放量

    Parameters
    ----------
    N
        5日平均成交量的倍数
    M
        流通股本的倍数

    """
    A2 = ts_delay(ts_mean(volume, 5), 1)
    C1 = volume / A2 > N
    C2 = volume / capital > M
    return C1 & C2


def 阶段缩量(volume: Expr, capital: Expr, N: int = 20, M: float = 0.02) -> Expr:
    """C111 阶段缩量

    Parameters
    ----------
    volume
        成交量
    capital
        流通股本

    Notes
    -----
    成交量与流通股本单位要一致，都为手，或者都为股

    """
    return ts_sum(volume, N) / capital <= M


def 阶段放量(volume: Expr, capital: Expr, N: int = 10, M: float = 2.0) -> Expr:
    """C112 阶段放量

    Parameters
    ----------
    volume
        成交量
    capital
        流通股本

    """
    return ts_sum(volume, N) / capital >= M


def 持续放量(volume: Expr, M: int = 5) -> Expr:
    """C113 持续放量

    Parameters
    ----------
    volume
        成交量

    """
    return ts_count(volume > ts_delay(volume, 1), M) == M


def 持续缩量(volume: Expr, M: int = 5) -> Expr:
    """C114 持续缩量

    Parameters
    ----------
    volume
        成交量

    """
    return ts_count(volume < ts_delay(volume, 1), M) == M


def 间隔放量(volume: Expr, N: int = 30, N1: float = 4.0, N2: float = 2.0, N3: int = 3) -> Expr:
    """C115 间隔放量

    Parameters
    ----------
    volume
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
    A = ts_mean(volume, N)  # 均量
    C1 = ts_max(A, N) < N1 * ts_min(A, N)  # 成交量最大与最小在一定范围内
    C2 = ts_count(volume > A * N2, N) > N3  # 成交量大于均量一定倍数
    return C1 & C2


def 放量上攻(close: Expr, volume: Expr, capital: Expr,
             N: float = 0.01, N1: int = 3, N2: float = 0.2, N3: int = 4) -> Expr:
    """C116 放量上攻

    Parameters
    ----------
    close
        复权收盘价
    volume
        成交量
    capital
        流通股本
    N
        涨幅
    N1, N2
        N1天内成交量大于N2*流通股本
    N3
        连续N3天满足涨幅

    """
    A = ts_returns(close, 1) >= N  # 涨幅大于N
    C1 = 阶段放量(volume, capital, N1, N2)
    C2 = 持续放量(volume, N3)
    C3 = ts_count(A, N3) == N3
    return C1 & C2 & C3


def 温和放量上攻(close: Expr, volume: Expr, capital: Expr, N: int = 5) -> Expr:
    """C117 温和放量上攻

    Parameters
    ----------
    close
        复权收盘价
    volume
        成交量
    capital
        流通股本
    N
        观察天数

    """
    A1 = close / ts_delay(close, 1)
    A2 = (A1 > 1) & (A1 < 1.03)  # {股价小幅上扬}
    B1 = volume / ts_delay(volume, 1)
    B2 = (B1 > 1) & (A1 < 2)  # {成交量小幅上扬}
    C1 = ts_mean(volume, N) / capital < 0.05  # 日成交量小于流通股本的5%
    C2 = ts_mean(A2 & B2, N) > 0.6
    return C1 & C2


def 突然放量(volume: Expr, N: int = 10, M: float = 3.0) -> Expr:
    """C118 突然放量

    Parameters
    ----------
    volume
        成交量
    N
        观察天数
    M
        倍数

    """
    return volume > ts_delay(ts_max(volume, N), 1) * M


def 平台整理(close: Expr, N: int = 30, N1: float = 0.05) -> Expr:
    """C119 平台整理

    Parameters
    ----------
    N
        天数
    N1
        幅度

    """
    return ts_max(close, N) / ts_min(close, N) <= 1 + N1


def 小步碎阳(open_: Expr, high: Expr, low: Expr, close: Expr, avg: Expr, turnover_ratio: Expr, N: int = 4) -> Expr:
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
    AA = (close > ts_delay(close, 1)) & (close > open_)
    A1 = ts_count(AA, N) == N
    A2 = close / ts_delay(close, N) < 1.05
    A3 = ts_WINNER_COST(high, low, avg, turnover_ratio, close, 0.5).struct[0] > 0.75

    return A1 & A2 & A3


def 突破长期盘整(high: Expr, low: Expr, close: Expr, N: int = 30, N1: int = 5) -> Expr:
    """C123 突破长期盘整

    Parameters
    ----------
    N
        天数
    N1
        涨幅

    """
    HH = ts_max(high, N)
    C0 = HH / ts_min(low, N)
    C1 = ts_delay(C0, 1) <= N1 + 1
    C2 = close > ts_delay(HH, 1)

    return C1 & C2


def N天内出现以涨停收盘(收盘涨停: Expr, N: int = 10) -> Expr:
    """C128 N天内出现以涨停收盘

    Parameters
    ----------
    N
        天数

    """
    return ts_count(收盘涨停, N) > 0


def N天内出现涨停(最高涨停: Expr, N: int = 20) -> Expr:
    """C129 N天内出现涨停

    Parameters
    ----------
    N
        天数

    """
    return ts_count(最高涨停, N) > 0


def N天内出现涨停(收盘涨停: Expr, N: int = 100, M: int = 8) -> Expr:
    """C129 N天内出现涨停

    Parameters
    ----------
    N
        天数
    M
        涨停天数

    """
    return ts_count(收盘涨停, N) >= M


def 下跌多日再放量上涨(high: Expr, close: Expr, volume: Expr) -> Expr:
    """C131 下跌多日再放量上涨

    """
    A1 = ts_delay(close, 5) > ts_delay(close, 4)
    A2 = ts_delay(close, 4) > ts_delay(close, 3)
    A3 = ts_delay(close, 3) > ts_delay(close, 2)
    A4 = ts_delay(close, 2) > ts_delay(close, 1)
    A5 = (close > ts_delay(high, 1)) & (volume > ts_delay(volume, 1))
    return A1 & A2 & A3 & A4 & A5


def 跳空高开或低开(open_: Expr, high: Expr, low: Expr, close: Expr, N: float = 0.03) -> Expr:
    """C132 跳空高开或低开

    Parameters
    ----------
    N
        涨幅

    """
    if N > 0:
        A = (open_ > ts_delay(high, 1)) & (open_ / ts_delay(close, 1) > (1 + N))
        return A
    else:
        B = (open_ < ts_delay(low, 1)) & (open_ / ts_delay(close, 1) < (1 + N))
        return B


def 拉升后多日调整(close: Expr, ZF: float = 0.09, N: int = 3) -> Expr:
    """C133 拉升后多日调整

    Parameters
    ----------
    ZF
        涨幅
    N
        天数

    """
    C0 = close / ts_delay(close, 1) - 1
    C1 = ts_delay(close / ts_delay(close, 1), N) > ZF
    C2 = ts_count(C0 < 0, N) == N
    return C1 & C2


def 昨日底部十字星(open_: Expr, high: Expr, low: Expr, close: Expr, N: int = 60) -> Expr:
    """C134 昨日底部十字星

    Parameters
    ----------
    N
        天数

    """

    C1 = low <= ts_min(low, N)
    C2 = abs_(close - open_) / (high - low) <= 0.05
    C3 = high > low
    return ts_delay(C1 & C2 & C3, 1)


def 价量渐低后阳包阴(open_: Expr, close: Expr, volume: Expr) -> Expr:
    """C135 价量渐低后阳包阴
    """
    A1 = ts_delay(close, 4) > ts_delay(close, 3)
    A2 = ts_delay(close, 3) > ts_delay(close, 2)
    A3 = ts_delay(close, 2) > ts_delay(close, 1)
    B1 = ts_delay(volume, 3) > ts_delay(volume, 2)
    B2 = ts_delay(volume, 2) > ts_delay(volume, 1)
    AA1 = A1 & A2 & A3 & B1 & B2
    AA2 = (ts_delay(open_, 1) > ts_delay(close, 1)) & (close > ts_delay(open_, 1))
    AA3 = (close > open_) & (volume < ts_delay(volume, 1))

    return AA1 & AA2 & AA3
