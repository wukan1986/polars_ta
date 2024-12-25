from polars import Expr

from polars_ta import TA_EPSILON
from polars_ta.tdx.reference import COUNT
from polars_ta.wq.logical import not_


def CROSS(a: Expr, b: Expr) -> Expr:
    """上穿

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 1, 1, 2],
        'b': [None, -1, 0, 1, 2],
        'c': [None, 0, 0, 0, 0],
        'd': [None, False, False, True, True],
    }).with_columns(
        out1=CROSS(pl.col('a'), pl.col('c')),
        out2=CROSS(pl.col('b'), pl.col('c')),
        out3=CROSS(0, pl.col('b')),
        out4=CROSS(pl.col('d'), 0.5),
    )
    shape: (5, 8)
    ┌──────┬──────┬──────┬───────┬───────┬───────┬───────┬───────┐
    │ a    ┆ b    ┆ c    ┆ d     ┆ out1  ┆ out2  ┆ out3  ┆ out4  │
    │ ---  ┆ ---  ┆ ---  ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   │
    │ i64  ┆ i64  ┆ i64  ┆ bool  ┆ bool  ┆ bool  ┆ bool  ┆ bool  │
    ╞══════╪══════╪══════╪═══════╪═══════╪═══════╪═══════╪═══════╡
    │ null ┆ null ┆ null ┆ null  ┆ null  ┆ null  ┆ null  ┆ null  │
    │ -1   ┆ -1   ┆ 0    ┆ false ┆ false ┆ false ┆ null  ┆ false │
    │ 1    ┆ 0    ┆ 0    ┆ false ┆ true  ┆ false ┆ false ┆ false │
    │ 1    ┆ 1    ┆ 0    ┆ true  ┆ false ┆ true  ┆ false ┆ true  │
    │ 2    ┆ 2    ┆ 0    ┆ true  ┆ false ┆ false ┆ false ┆ false │
    └──────┴──────┴──────┴───────┴───────┴───────┴───────┴───────┘
    ```

    """
    c = a < b - TA_EPSILON
    d = abs(a - b) < TA_EPSILON
    e = a > b + TA_EPSILON
    # 1. 小于大于
    # 2. 小于等于大于
    # 3. 小于等于等于大于。中间等了两期，暂时不定义为上穿
    return e & (c.shift(1) | (d.shift(1) & c.shift(2)))


def DOWNNDAY(close: Expr, N: int) -> Expr:
    """返回周期数内是否连跌"""
    return NDAY(close.shift(), close, N)


def EVERY(condition: Expr, N: int) -> Expr:
    """一直存在"""
    return COUNT(condition, N) == N


def EXIST(condition: Expr, N: int) -> Expr:
    """是否存在"""
    return COUNT(condition, N) > 0


def EXISTR(condition: Expr, a: int, b: int) -> Expr:
    """从前a日到前b日内是否存在"""
    return EXIST(condition, a - b).shift(b)


def LAST(condition: Expr, a: int, b: int) -> Expr:
    """从前a日到前b日内一直存在"""
    return EVERY(condition, a - b).shift(b)


def LONGCROSS(a: Expr, b: Expr, N: int) -> Expr:
    """两条线维持一定周期后交叉"""
    return CROSS(a, b) & EVERY(a < (b - TA_EPSILON), N).shift(1)


def NDAY(close: Expr, open_: Expr, N: int) -> Expr:
    """返回是否持续存在X>Y"""
    return EVERY(close > (open_ + TA_EPSILON), N)


def NOT(condition: Expr) -> Expr:
    """求逻辑非"""
    return not_(condition)


def UPNDAY(close: Expr, N: int) -> Expr:
    """返回周期数内是否连涨"""
    return NDAY(close, close.shift(), N)


# Eastmoney has these two functions
# 东方财富中有此两函数
ALL = EVERY
ANY = EXIST
