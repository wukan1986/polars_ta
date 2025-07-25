from datetime import time, timedelta

from polars import Expr, when


def FROMOPEN(t: Expr) -> Expr:
    """返回当前时刻距开盘有多少分钟

    范围0~240,开盘前为0，10点为31

    Examples
    --------
    from datetime import datetime

    import polars as pl

    from polars_ta.tdx.times import FROMOPEN, FROMOPEN1

    df = pl.DataFrame({'datetime': [
        datetime(2025, 1, 1, 0, 0),
        datetime(2025, 1, 1, 9, 25),
        datetime(2025, 1, 1, 9, 30, 57),
        datetime(2025, 1, 1, 9, 31),
        datetime(2025, 1, 1, 10, 0),
        datetime(2025, 1, 1, 13, 0),
    ]})

    df = df.with_columns(
        FROMOPEN=FROMOPEN(pl.col('datetime')),
        FROMOPEN1=FROMOPEN_1(pl.col('datetime'), 0),
        FROMOPEN2=FROMOPEN_1(pl.col('datetime'), 60),
    )

    shape: (6, 4)
    ┌─────────────────────┬──────────┬───────────┬───────────┐
    │ datetime            ┆ FROMOPEN ┆ FROMOPEN1 ┆ FROMOPEN2 │
    │ ---                 ┆ ---      ┆ ---       ┆ ---       │
    │ datetime[μs]        ┆ i64      ┆ i64       ┆ i64       │
    ╞═════════════════════╪══════════╪═══════════╪═══════════╡
    │ 2025-01-01 00:00:00 ┆ 0        ┆ 240       ┆ 240       │
    │ 2025-01-01 09:25:00 ┆ 0        ┆ 1         ┆ 1         │
    │ 2025-01-01 09:30:57 ┆ 1        ┆ 1         ┆ 2         │
    │ 2025-01-01 09:31:00 ┆ 2        ┆ 2         ┆ 3         │
    │ 2025-01-01 10:00:00 ┆ 31       ┆ 31        ┆ 32        │
    │ 2025-01-01 13:00:00 ┆ 121      ┆ 121       ┆ 122       │
    └─────────────────────┴──────────┴───────────┴───────────┘

    """
    am = (t.dt.time() - time(9, 29)).dt.total_minutes().clip(0, 120)
    pm = (t.dt.time() - time(12, 59)).dt.total_minutes().clip(0, 120)
    return am + pm


def FROMOPEN_1(t: Expr, offset: int) -> Expr:
    """返回当前时刻距开盘有多少分钟。范围1~240

    用于计算量比
    1. 竞价量比，分母应当为1
    2. 日线数据0~8点时，返回240
    3. 日线数据9点时，返回1

    Parameters
    ----------
    t : Expr
        时间列
    offset : int
        偏移量，单位秒

    Notes
    -----
    每根K线结束时，标签是当前时间的50多秒，而结束时时间已经到下以分钟了，所以建议加60秒

    """
    return when(t.dt.time() >= time(8, 45)).then(FROMOPEN(t + timedelta(seconds=offset)).clip(1, 240)).otherwise(240)
