from polars import Expr, Struct, Field, Int64

from polars_ta.tdx._nb import roll_avedev, _up_stat
from polars_ta.utils.numba_ import batches_i1_o1, batches_i1_o2
from polars_ta.wq.time_series import ts_corr as RELATE  # noqa
from polars_ta.wq.time_series import ts_covariance as COVAR  # noqa
from polars_ta.wq.time_series import ts_std_dev as _ts_std_dev


def AVEDEV(close: Expr, timeperiod: int = 5) -> Expr:
    """mean absolute deviation
    平均绝对偏差"""
    return close.map_batches(lambda x1: batches_i1_o1(x1.to_numpy(), roll_avedev, timeperiod))


def DEVSQ(close: Expr, timeperiod: int = 5) -> Expr:
    raise


def SLOPE(close: Expr, timeperiod: int = 5) -> Expr:
    raise


def STD(close: Expr, timeperiod: int = 5) -> Expr:
    """std dev with ddof = 1
    估算标准差"""
    return _ts_std_dev(close, timeperiod, 1)


def STDDEV(close: Expr, timeperiod: int = 5) -> Expr:
    """标准偏差?"""
    raise


def STDP(close: Expr, timeperiod: int = 5) -> Expr:
    """std dev with ddof = 0
    总体标准差"""
    return _ts_std_dev(close, timeperiod, 0)


def VAR(close: Expr, timeperiod: int = 5) -> Expr:
    return close.rolling_var(timeperiod, ddof=1)


def VARP(close: Expr, timeperiod: int = 5) -> Expr:
    return close.rolling_var(timeperiod, ddof=0)


def ts_up_stat(x: Expr) -> Expr:
    """T天N板统计，与通达信结果一样，最简为5天2板

    Parameters
    ----------
    x: Expr
        布尔序列，True表示涨停

    Returns
    -------
    Expr
        1. T天
        2. N板
        3. 离上次涨停距离

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [False, True, True, False, True, False, False, False, False, False],
    }).with_columns(
        out=ts_up_stat(pl.col('a'))
    )
    ┌───────┬───────────┐
    │ a     ┆ out       │
    │ ---   ┆ ---       │
    │ bool  ┆ struct[3] │
    ╞═══════╪═══════════╡
    │ false ┆ {0,0,0}   │
    │ true  ┆ {1,1,0}   │
    │ true  ┆ {2,2,0}   │
    │ false ┆ {3,2,1}   │
    │ true  ┆ {4,3,0}   │
    │ false ┆ {5,3,1}   │
    │ false ┆ {6,3,2}   │
    │ false ┆ {7,3,3}   │
    │ false ┆ {0,0,4}   │
    │ false ┆ {0,0,5}   │
    └───────┴───────────┘
    ```

    """
    dtype = Struct([Field(f"column_{i}", Int64) for i in range(3)])
    return x.map_batches(lambda x1: batches_i1_o2(x1.to_numpy(), _up_stat), return_dtype=dtype)
