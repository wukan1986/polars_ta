"""

由于标签的定义比较灵活，所以以下代码主要用于参考

Notes
-----
标签都是未来数据,在机器学习中，只能用于`y`,不能用于`X`。

References
----------
https://mp.weixin.qq.com/s/XtgYezFsslOfW-QyIMr0VA
https://github.com/Rachnog/Advanced-Deep-Trading/blob/master/bars-labels-diff/Labeling.ipynb

"""
from polars import Expr, struct

from polars_ta.labels._nb import _triple_barrier
from polars_ta.utils.numba_ import batches_i2_o1, struct_to_numpy
from polars_ta.wq import cut, ts_delay, ts_log_diff, log


def ts_log_return(close: Expr, n: int = 5) -> Expr:
    """将未来数据当成卖出价后移到买入价位置，计算对数收益率

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 10, 11, 12, 9, 12, 13],
    }).with_columns(
        out1=ts_log_return(pl.col('a'), 3),
        out2=_ts_log_return(pl.col('a'), 3),
    )

    shape: (7, 3)
    ┌──────┬───────────┬───────────┐
    │ a    ┆ out1      ┆ out2      │
    │ ---  ┆ ---       ┆ ---       │
    │ i64  ┆ f64       ┆ f64       │
    ╞══════╪═══════════╪═══════════╡
    │ null ┆ null      ┆ null      │
    │ 10   ┆ -0.105361 ┆ -0.105361 │
    │ 11   ┆ 0.087011  ┆ 0.087011  │
    │ 12   ┆ 0.080043  ┆ 0.080043  │
    │ 9    ┆ null      ┆ null      │
    │ 12   ┆ null      ┆ null      │
    │ 13   ┆ null      ┆ null      │
    └──────┴───────────┴───────────┘
    ```

    """
    # return (close.shift(-n) / close).log()
    return log(ts_delay(close, -n) / close)


def _ts_log_return(close: Expr, n: int = 5) -> Expr:
    """计算对数收益率，但将结果后移

    如果打标签方式复杂，这种最终结果后移的方法更方便
    """
    # return (close / close.shift(n)).log().shift(-n)
    return ts_delay(ts_log_diff(close, n), -n)


def ts_simple_return(close: Expr, n: int = 5, threshold: float = 0.0, *more_threshold) -> Expr:
    """简单收益率标签。支持二分类、三分类等。对收益率使用`cut`进行分类
    
    Parameters
    ----------
    close
    n:int
        未来n天
    threshold:float
        收益率阈值，小于该值为0，大于等于该值为1
    more_threshold:float
        更多的阈值，用于三分类等。小于该值为1，大于等于该值为2，以此类推

    Returns
    -------
    Expr
        标签列, 类型为UInt32, 取值为0, 1, 2, ...

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 10., 9.99, 9., 10., 11., 11.],
    }).with_columns(
        out1=label_simple_return(pl.col('a'), 1, 0),
        out2=label_simple_return(pl.col('a'), 1, -0.001, 0.001),
    )

    shape: (7, 3)
    ┌──────┬──────┬──────┐
    │ a    ┆ out1 ┆ out2 │
    │ ---  ┆ ---  ┆ ---  │
    │ f64  ┆ u32  ┆ u32  │
    ╞══════╪══════╪══════╡
    │ null ┆ null ┆ null │
    │ 10.0 ┆ 0    ┆ 0    │
    │ 9.99 ┆ 0    ┆ 0    │
    │ 9.0  ┆ 1    ┆ 2    │
    │ 10.0 ┆ 1    ┆ 2    │
    │ 11.0 ┆ 0    ┆ 1    │
    │ 11.0 ┆ null ┆ null │
    └──────┴──────┴──────┘

    """
    return cut(close.pct_change(n).shift(-n), threshold, *more_threshold)


def ts_triple_barrier(close: Expr, high: Expr, low: Expr, d: int = 5, take_profit: float = 0.1, stop_loss: float = 0.05) -> Expr:
    """三重障碍打标法

    Parameters
    ----------
    close:Expr
        收盘价
    high:Expr
        最高价
    low:Expr
        最低价
    d:int
        时间窗口
    take_profit:float
        止盈比例
    stop_loss:float
        止损比例

    Returns
    -------
    Expr
        标签列。取值为-1止损, 1止盈，0时间到期

    Notes
    -----
    1. `high`, `low`在粗略情况下可用`close`代替
    2. 时间到期时，根据盈亏返回不同的标签

    Examples
    --------
    ```python
    df = pl.DataFrame({
        "close": [np.nan, 1, 1, 1.0],
        "high": [np.nan, 1, 1.1, 1],
        "low": [np.nan, 1, 1, 0.95],
    }).with_columns(
        out=ts_triple_barrier(pl.col("close"), pl.col("high"), pl.col("low"), 2, 0.1, 0.05)
    )

    shape: (4, 4)
    ┌───────┬──────┬──────┬──────┐
    │ close ┆ high ┆ low  ┆ out  │
    │ ---   ┆ ---  ┆ ---  ┆ ---  │
    │ f64   ┆ f64  ┆ f64  ┆ f64  │
    ╞═══════╪══════╪══════╪══════╡
    │ NaN   ┆ NaN  ┆ NaN  ┆ null │
    │ 1.0   ┆ 1.0  ┆ 1.0  ┆ 1.0  │
    │ 1.0   ┆ 1.1  ┆ 1.0  ┆ -1.0 │
    │ 1.0   ┆ 1.0  ┆ 0.95 ┆ null │
    └───────┴──────┴──────┴──────┘
    ```

    """
    return struct([close, high, low]).map_batches(lambda xx: batches_i2_o1(struct_to_numpy(xx, 3), _triple_barrier, d, take_profit, stop_loss))
