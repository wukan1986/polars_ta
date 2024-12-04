"""
通过`import`直接导入或更名的函数

```python
from polars_ta.wq.arithmetic import add as ADD  # noqa
from polars_ta.wq.arithmetic import divide as DIV  # noqa
from polars_ta.wq.arithmetic import multiply as MUL  # noqa
from polars_ta.wq.arithmetic import subtract as SUB  # noqa
from polars_ta.wq.time_series import ts_max as MAX  # noqa
from polars_ta.wq.time_series import ts_min as MIN  # noqa
from polars_ta.wq.time_series import ts_sum as SUM  # noqa
```

"""
from polars import Expr

from polars_ta.wq.arithmetic import add as ADD  # noqa
from polars_ta.wq.arithmetic import divide as DIV  # noqa
from polars_ta.wq.arithmetic import multiply as MUL  # noqa
from polars_ta.wq.arithmetic import subtract as SUB  # noqa
from polars_ta.wq.time_series import ts_arg_max
from polars_ta.wq.time_series import ts_arg_min
from polars_ta.wq.time_series import ts_max as MAX  # noqa
from polars_ta.wq.time_series import ts_min as MIN  # noqa
from polars_ta.wq.time_series import ts_sum as SUM  # noqa


def MAXINDEX(close: Expr, timeperiod: int = 30) -> Expr:
    """

    Notes
    -----
    Comparing to `ts_arg_max` this also marks the abs. position of the max value

    与ts_arg_max的区别是，标记了每个区间最大值的绝对位置，可用来画图标记


    Examples
    --------
    ```python
    from polars_ta.ta import MAXINDEX as ta_MAXINDEX
    from polars_ta.talib import MAXINDEX as talib_MAXINDEX
    from polars_ta.wq import ts_arg_max

    df = pl.DataFrame({
        'a': [6, 2, 8, 5, 9, 4],
    }).with_columns(
        out1=ts_arg_max(pl.col('a'), 3),
        out2=ta_MAXINDEX(pl.col('a'), 3),
        out3=talib_MAXINDEX(pl.col('a'), 3),
    )
    shape: (6, 4)
    ┌─────┬──────┬──────┬──────┐
    │ a   ┆ out1 ┆ out2 ┆ out3 │
    │ --- ┆ ---  ┆ ---  ┆ ---  │
    │ i64 ┆ u16  ┆ i64  ┆ i32  │
    ╞═════╪══════╪══════╪══════╡
    │ 6   ┆ null ┆ null ┆ 0    │
    │ 2   ┆ null ┆ null ┆ 0    │
    │ 8   ┆ 0    ┆ 2    ┆ 2    │
    │ 5   ┆ 1    ┆ 2    ┆ 2    │
    │ 9   ┆ 0    ┆ 4    ┆ 4    │
    │ 4   ┆ 1    ┆ 4    ┆ 4    │
    └─────┴──────┴──────┴──────┘
    ```
    """
    a = close.cum_count()
    b = ts_arg_max(close, timeperiod)
    return a - b - 1


def MININDEX(close: Expr, timeperiod: int = 30) -> Expr:
    """



    Examples
    --------
    ```python
    from polars_ta.ta import MININDEX as ta_MININDEX
    from polars_ta.talib import MININDEX as talib_MININDEX
    from polars_ta.wq import ts_arg_min

    df = pl.DataFrame({
        'a': [6, 2, 8, 5, 9, 4],

    }).with_columns(
        out1=ts_arg_min(pl.col('a'), 3),
        out2=ta_MININDEX(pl.col('a'), 3),
        out3=talib_MININDEX(pl.col('a'), 3),
    )
    shape: (6, 4)
    ┌─────┬──────┬──────┬──────┐
    │ a   ┆ out1 ┆ out2 ┆ out3 │
    │ --- ┆ ---  ┆ ---  ┆ ---  │
    │ i64 ┆ u16  ┆ i64  ┆ i32  │
    ╞═════╪══════╪══════╪══════╡
    │ 6   ┆ null ┆ null ┆ 0    │
    │ 2   ┆ null ┆ null ┆ 0    │
    │ 8   ┆ 1    ┆ 1    ┆ 1    │
    │ 5   ┆ 2    ┆ 1    ┆ 1    │
    │ 9   ┆ 1    ┆ 3    ┆ 3    │
    │ 4   ┆ 0    ┆ 5    ┆ 5    │
    └─────┴──────┴──────┴──────┘
    ```
    """
    a = close.cum_count()
    b = ts_arg_min(close, timeperiod)
    return a - b - 1
