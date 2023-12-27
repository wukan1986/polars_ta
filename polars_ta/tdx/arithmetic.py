import polars as pl

from polars_ta.wq.arithmetic import abs_ as ABS  # noqa
from polars_ta.wq.arithmetic import add as ADD  # noqa
from polars_ta.wq.arithmetic import ceiling as CEILING  # noqa
from polars_ta.wq.arithmetic import cos as COS  # noqa
from polars_ta.wq.arithmetic import exp as EXP  # noqa
from polars_ta.wq.arithmetic import floor as FLOOR  # noqa
from polars_ta.wq.arithmetic import fraction as FRACPART  # noqa
from polars_ta.wq.arithmetic import log as LN  # noqa # 自然对数
from polars_ta.wq.arithmetic import log10 as LOG  # noqa # 10为底的对数
from polars_ta.wq.arithmetic import max_ as MAX  # noqa
from polars_ta.wq.arithmetic import min_ as MIN  # noqa
from polars_ta.wq.arithmetic import mod as MOD  # noqa
from polars_ta.wq.arithmetic import power as POW  # noqa
from polars_ta.wq.arithmetic import reverse as REVERSE  # noqa
from polars_ta.wq.arithmetic import round_ as _round  # noqa
from polars_ta.wq.arithmetic import sign as SIGN  # noqa
from polars_ta.wq.arithmetic import sin as SIN  # noqa
from polars_ta.wq.arithmetic import sqrt as SQRT  # noqa
from polars_ta.wq.arithmetic import subtract as SUB  # noqa
from polars_ta.wq.arithmetic import tan as TAN  # noqa
from polars_ta.wq.arithmetic import truncate as INTPART  # noqa
from polars_ta.wq.transformational import arc_cos as ACOS  # noqa
from polars_ta.wq.transformational import arc_sin as ASIN  # noqa
from polars_ta.wq.transformational import arc_tan as ATAN  # noqa

SGN = SIGN


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