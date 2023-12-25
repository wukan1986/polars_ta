import polars as pl

from polars_ta.ta.overlap import EMA as _ema
from polars_ta.ta.overlap import SMA as MA  # noqa
from polars_ta.ta.volatility import TRANGE as TR  # noqa
from polars_ta.wq.cross_sectional import rank as RANK  # noqa
from polars_ta.wq.time_series import ts_arg_max as HHVBARS  # noqa
from polars_ta.wq.time_series import ts_arg_min as LLVBARS  # noqa
from polars_ta.wq.time_series import ts_count as COUNT  # noqa
from polars_ta.wq.time_series import ts_decay_linear as WMA  # noqa
from polars_ta.wq.time_series import ts_delay as REF  # noqa
from polars_ta.wq.time_series import ts_delta as DIFF  # noqa
from polars_ta.wq.time_series import ts_max as HHV  # noqa
from polars_ta.wq.time_series import ts_min as LLV  # noqa
from polars_ta.wq.time_series import ts_product as MULAR  # noqa
from polars_ta.wq.time_series import ts_sum as SUM  # noqa


def DMA(close: pl.Expr, alpha: float = 0.5) -> pl.Expr:
    """DMA(X,A),求X的动态移动平均.
算法:Y=A*X+(1-A)*Y',其中Y'表示上一周期Y值,A必须大于0且小于1.A支持变量"""
    return close.ewm_mean(alpha=alpha, adjust=False, min_periods=1)


def EMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """EMA(X,N):X的N日指数移动平均.算法:Y=(X*2+Y'*(N-1))/(N+1)
 EMA(X,N)相当于SMA(X,N+1,2),N支持变量"""
    return _ema(close, timeperiod)


def EXPMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    return _ema(close, timeperiod)


def EXPMEMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """EXPMEMA(X,M),X的M日指数平滑移动平均。EXPMEMA同EMA(即EXPMA)的差别在于他的起始值为一平滑值

    Notes
    -----
    等价于talib.EMA，由于比EMA慢，少用

    """
    sma = MA(close, timeperiod)
    x = pl.when(close.cum_count() < timeperiod).then(sma).otherwise(close)
    return x.ewm_mean(span=timeperiod, adjust=False, min_periods=1)


def MEMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """MEMA(X,N):X的N日平滑移动平均,如Y=(X+Y'*(N-1))/N
 MEMA(X,N)相当于SMA(X,N,1)"""
    raise


def RANGE(a: pl.Expr, b: pl.Expr, c: pl.Expr) -> pl.Expr:
    """A在B和C范围之间,B<A<C."""
    return (b < a) & (a < c)


def SMA(X: pl.Expr, N: int, M: int = 1) -> pl.Expr:
    """用法:SMA(X,N,M),X的N日移动平均,M为权重,若Y=SMA(X,N,M)则Y=(X*M+Y'*(N-M))/N"""
    return X.ewm_mean(alpha=M / N, adjust=False, min_periods=1)


def TMA(close: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """TMA(X,A,B),A和B必须小于1,算法	Y=(A*Y'+B*X),其中Y'表示上一周期Y值.初值为X"""
    raise
