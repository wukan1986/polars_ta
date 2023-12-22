import polars as pl

from polars_ta.wq.arithmetic import abs_ as ABS
# from polars_ta.wq.arithmetic import add as ADD
from polars_ta.wq.arithmetic import ceiling as CEILING
from polars_ta.wq.arithmetic import exp as EXP
from polars_ta.wq.arithmetic import floor as FLOOR
from polars_ta.wq.arithmetic import fraction as FRACPART
from polars_ta.wq.arithmetic import log as LN  # 自然对数
from polars_ta.wq.arithmetic import log10 as LOG  # 10为底的对数
from polars_ta.wq.arithmetic import max_ as MAX
from polars_ta.wq.arithmetic import min_ as MIN
from polars_ta.wq.arithmetic import mod as MOD
from polars_ta.wq.arithmetic import power as POW
from polars_ta.wq.arithmetic import reverse as REVERSE
from polars_ta.wq.arithmetic import round_ as _round
from polars_ta.wq.arithmetic import sign as SIGN
from polars_ta.wq.arithmetic import sqrt as SQRT
# from polars_ta.wq.arithmetic import subtract as SUB
from polars_ta.wq.arithmetic import truncate as INTPART
from polars_ta.wq.transformational import arc_cos as ACOS
from polars_ta.wq.transformational import arc_sin as ASIN
from polars_ta.wq.transformational import arc_tan as ATAN

_ = ABS, CEILING, EXP, FLOOR, FRACPART, LN, LOG, MAX, MIN, MOD, POW, REVERSE, SIGN, SQRT, INTPART
_ = ACOS, ASIN, ATAN


def ROUND(x: pl.Expr) -> pl.Expr:
    """Round input to closest integer."""
    return _round(x, 0)


def ROUND2(x: pl.Expr, decimals: int = 0) -> pl.Expr:
    """Round input to closest integer."""
    return _round(x, decimals)


def BETWEEN(a: pl.Expr, b: pl.Expr, c: pl.Expr) -> pl.Expr:
    """BETWEEN(A,B,C)表示A处于B和C之间时返回1(B<=A<=C或C<=A<=B),否则返回0"""
    x1 = (b <= a) & (a <= c)
    x2 = (c <= a) & (a <= b)
    return x1 | x2
