import polars_ols as pls
from polars import Expr, UInt16, struct, when, Struct, Field, Float64, Boolean, UInt32
from polars import rolling_corr, rolling_cov
from polars_ols import RollingKwargs

from polars_ta.utils.numba_ import batches_i1_o1, batches_i2_o1, batches_i2_o2
from polars_ta.utils.pandas_ import roll_kurt, roll_rank
from polars_ta.wq._nb import roll_argmax, roll_argmin, roll_prod, roll_co_kurtosis, roll_co_skewness, roll_moment, roll_partial_corr, roll_triple_corr, _cum_prod_by, _cum_sum_by, _signals_to_size, _cum_sum_reset, _sum_split_by, roll_decay_linear, roll_decay_exp_window


def ts_arg_max(x: Expr, d: int = 5, reverse: bool = True) -> Expr:
    """Returns the relative index of the max value in the time series for the past d days.
    If the current day has the max value for the past d days, it returns 0.
    If previous day has the max value for the past d days, it returns 1.

    Parameters
    ----------
    x
    d
    reverse
        反向

    See Also
    --------
    ts_arg_min

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [6, 2, 8, 5, 9, 4][::-1],
    }).with_columns(
        out=ts_arg_max(pl.col('a'), 6),
    )
    shape: (6, 2)
    ┌─────┬──────┐
    │ a   ┆ out  │
    │ --- ┆ ---  │
    │ i64 ┆ u16  │
    ╞═════╪══════╡
    │ 4   ┆ null │
    │ 9   ┆ null │
    │ 5   ┆ null │
    │ 8   ┆ null │
    │ 2   ┆ null │
    │ 6   ┆ 4    │
    └─────┴──────┘
    ```

    References
    ----------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#ts_arg_maxx-d

    """
    return x.map_batches(lambda x1: batches_i1_o1(x1.to_numpy(), roll_argmax, d, reverse, dtype=UInt16))


def ts_arg_min(x: Expr, d: int = 5, reverse: bool = True) -> Expr:
    """

    Parameters
    ----------
    x
    d
    reverse
        反向

    See Also
    --------
    ts_arg_max

    References
    ----------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#ts_arg_minx-d

    """
    return x.map_batches(lambda x1: batches_i1_o1(x1.to_numpy(), roll_argmin, d, reverse, dtype=UInt16))


def ts_co_kurtosis(x: Expr, y: Expr, d: int = 5, ddof: int = 0) -> Expr:
    return struct([x, y]).map_batches(lambda xx: batches_i2_o1([xx.struct[i].to_numpy() for i in range(2)], roll_co_kurtosis, d))


def ts_co_skewness(x: Expr, y: Expr, d: int = 5, ddof: int = 0) -> Expr:
    return struct([x, y]).map_batches(lambda xx: batches_i2_o1([xx.struct[i].to_numpy() for i in range(2)], roll_co_skewness, d))


def ts_corr(x: Expr, y: Expr, d: int = 5, ddof: int = 1) -> Expr:
    """rolling correlation between two columns

    时序滚动相关系数

    Parameters
    ----------
    x
    y
    d
    ddof
        自由度

    Notes
    -----
    x、y不区分先后

    """
    return rolling_corr(x, y, window_size=d, ddof=ddof)


def ts_count(x: Expr, d: int = 30) -> Expr:
    """时序滚动计数

    Parameters
    ----------
    x
    d

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2, 3],
        'b': [None, True, True, True, False, False],
    }).with_columns(
        out1=ts_count(pl.col('a'), 3),
        out2=ts_count(pl.col('b'), 3),
    )

    shape: (6, 4)
    ┌──────┬───────┬──────┬──────┐
    │ a    ┆ b     ┆ out1 ┆ out2 │
    │ ---  ┆ ---   ┆ ---  ┆ ---  │
    │ i64  ┆ bool  ┆ u32  ┆ u32  │
    ╞══════╪═══════╪══════╪══════╡
    │ null ┆ null  ┆ null ┆ null │
    │ -1   ┆ true  ┆ null ┆ null │
    │ 0    ┆ true  ┆ null ┆ null │
    │ 1    ┆ true  ┆ 2    ┆ 3    │
    │ 2    ┆ false ┆ 2    ┆ 2    │
    │ 3    ┆ false ┆ 3    ┆ 1    │
    └──────┴───────┴──────┴──────┘
    ```

    """
    return x.cast(Boolean).cast(UInt32).rolling_sum(d)


def ts_count_nans(x: Expr, d: int = 5) -> Expr:
    """时序滚动统计nan出现次数

    Parameters
    ----------
    x
    d

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, float('nan'), -1, 0, 1, 2, 3],
    }).with_columns(
        out1=ts_count_nans(pl.col('a'), 3),
    )

    shape: (7, 2)
    ┌──────┬──────┐
    │ a    ┆ out1 │
    │ ---  ┆ ---  │
    │ f64  ┆ u32  │
    ╞══════╪══════╡
    │ null ┆ null │
    │ NaN  ┆ null │
    │ -1.0 ┆ null │
    │ 0.0  ┆ 1    │
    │ 1.0  ┆ 0    │
    │ 2.0  ┆ 0    │
    │ 3.0  ┆ 0    │
    └──────┴──────┘
    ```

    """
    return x.is_nan().cast(UInt32).rolling_sum(d)


def ts_count_nulls(x: Expr, d: int = 5) -> Expr:
    """时序滚动统计null出现次数

    Parameters
    ----------
    x
    d

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2, 3],
        'b': [None, True, True, True, False, False],
    }).with_columns(
        out1=ts_count_nulls(pl.col('a'), 3),
        out2=ts_count_nulls(pl.col('b'), 3),
    )
    shape: (6, 4)
    ┌──────┬───────┬──────┬──────┐
    │ a    ┆ b     ┆ out1 ┆ out2 │
    │ ---  ┆ ---   ┆ ---  ┆ ---  │
    │ i64  ┆ bool  ┆ u32  ┆ u32  │
    ╞══════╪═══════╪══════╪══════╡
    │ null ┆ null  ┆ null ┆ null │
    │ -1   ┆ true  ┆ null ┆ null │
    │ 0    ┆ true  ┆ 1    ┆ 1    │
    │ 1    ┆ true  ┆ 0    ┆ 0    │
    │ 2    ┆ false ┆ 0    ┆ 0    │
    │ 3    ┆ false ┆ 0    ┆ 0    │
    └──────┴───────┴──────┴──────┘
    ```

    """
    return x.is_null().cast(UInt32).rolling_sum(d)


def ts_covariance(x: Expr, y: Expr, d: int = 5, ddof: int = 1) -> Expr:
    """rolling covariance between two columns

    时序协方差

    Parameters
    ----------
    x
    y
    d
    ddof
        自由度

    Notes
    -----
    x、y不区分先后

    """
    return rolling_cov(x, y, window_size=d, ddof=ddof)


def ts_cum_count(x: Expr) -> Expr:
    """时序累计计数

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, None, -1, 0, 1, 2],
        'b': [None, None, True, True, True, False],
    }).with_columns(
        out1=ts_cum_count(pl.col('a')),
        out2=ts_cum_count(pl.col('b')),
    )
    shape: (6, 4)
    ┌──────┬───────┬──────┬──────┐
    │ a    ┆ b     ┆ out1 ┆ out2 │
    │ ---  ┆ ---   ┆ ---  ┆ ---  │
    │ i64  ┆ bool  ┆ u32  ┆ u32  │
    ╞══════╪═══════╪══════╪══════╡
    │ null ┆ null  ┆ 0    ┆ 0    │
    │ null ┆ null  ┆ 0    ┆ 0    │
    │ -1   ┆ true  ┆ 1    ┆ 1    │
    │ 0    ┆ true  ┆ 2    ┆ 2    │
    │ 1    ┆ true  ┆ 3    ┆ 3    │
    │ 2    ┆ false ┆ 4    ┆ 4    │
    └──────┴───────┴──────┴──────┘
    ```

    """
    return x.cum_count()


def ts_cum_max(x: Expr) -> Expr:
    """时序累计最大

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, None, -1, 0, 2, 1],
        'b': [None, None, True, False, False, True],
    }).with_columns(
        out1=ts_cum_max(pl.col('a')),
        out2=ts_cum_max(pl.col('b')),
    )
    shape: (6, 4)
    ┌──────┬───────┬──────┬──────┐
    │ a    ┆ b     ┆ out1 ┆ out2 │
    │ ---  ┆ ---   ┆ ---  ┆ ---  │
    │ i64  ┆ bool  ┆ i64  ┆ bool │
    ╞══════╪═══════╪══════╪══════╡
    │ null ┆ null  ┆ null ┆ null │
    │ null ┆ null  ┆ null ┆ null │
    │ -1   ┆ true  ┆ -1   ┆ true │
    │ 0    ┆ false ┆ 0    ┆ true │
    │ 2    ┆ false ┆ 2    ┆ true │
    │ 1    ┆ true  ┆ 2    ┆ true │
    └──────┴───────┴──────┴──────┘
    ```

    """
    return x.cum_max()


def ts_cum_min(x: Expr) -> Expr:
    """时序累计最小

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, None, -1, 0, -2, 1],
        'b': [None, None, True, False, False, True],
    }).with_columns(
        out1=ts_cum_min(pl.col('a')),
        out2=ts_cum_min(pl.col('b')),
    )
    shape: (6, 4)
    ┌──────┬───────┬──────┬───────┐
    │ a    ┆ b     ┆ out1 ┆ out2  │
    │ ---  ┆ ---   ┆ ---  ┆ ---   │
    │ i64  ┆ bool  ┆ i64  ┆ bool  │
    ╞══════╪═══════╪══════╪═══════╡
    │ null ┆ null  ┆ null ┆ null  │
    │ null ┆ null  ┆ null ┆ null  │
    │ -1   ┆ true  ┆ -1   ┆ true  │
    │ 0    ┆ false ┆ -1   ┆ false │
    │ -2   ┆ false ┆ -2   ┆ false │
    │ 1    ┆ true  ┆ -2   ┆ false │
    └──────┴───────┴──────┴───────┘
    ```
    """
    return x.cum_min()


def ts_cum_prod(x: Expr) -> Expr:
    """时序累乘"""
    return x.cum_prod()


def ts_cum_sum(x: Expr) -> Expr:
    """时序累加"""
    return x.cum_sum()


def ts_cum_sum_reset(x: Expr) -> Expr:
    """时序累加。遇到0、nan、相反符号时重置

    Examples
    --------

    ```python
    df = pl.DataFrame({
        'a': [1, 0, 1, 2, None, 3, -2, -3],
    }).with_columns(
        A=ts_cum_sum_reset(pl.col('a'))
    )

    shape: (8, 2)
    ┌──────┬──────┐
    │ a    ┆ A    │
    │ ---  ┆ ---  │
    │ i64  ┆ f64  │
    ╞══════╪══════╡
    │ 1    ┆ 1.0  │
    │ 0    ┆ 0.0  │
    │ 1    ┆ 1.0  │
    │ 2    ┆ 3.0  │
    │ null ┆ 0.0  │
    │ 3    ┆ 3.0  │
    │ -2   ┆ -2.0 │
    │ -3   ┆ -5.0 │
    └──────┴──────┘

    ```

    """
    return x.map_batches(lambda x1: batches_i1_o1(x1.to_numpy().astype(float), _cum_sum_reset))


def ts_decay_exp_window(x: Expr, d: int = 30, factor: float = 1.0) -> Expr:
    """指数衰减移动平均

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 6, 5, 4, 5, 30],
    }).with_columns(
        out1=ts_decay_linear(pl.col('a'), 5),
        out2=ts_decay_exp_window(pl.col('a'), 5, 0.5),
    )
    shape: (6, 3)
    ┌──────┬──────┬───────────┐
    │ a    ┆ out1 ┆ out2      │
    │ ---  ┆ ---  ┆ ---       │
    │ i64  ┆ f64  ┆ f64       │
    ╞══════╪══════╪═══════════╡
    │ null ┆ null ┆ null      │
    │ 6    ┆ null ┆ null      │
    │ 5    ┆ null ┆ null      │
    │ 4    ┆ null ┆ null      │
    │ 5    ┆ null ┆ null      │
    │ 30   ┆ 13.2 ┆ 17.806452 │
    └──────┴──────┴───────────┘
    ```

    Parameters
    ----------
    x
    d
    factor
        衰减系数

    References
    ----------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#ts_decay_exp_windowx-d-factor-10-nan-true

    """
    # y = arange(d - 1, -1, step=-1, eager=False)
    # weights = repeat(factor, d, eager=True).pow(y)
    # print(weights)
    # return x.rolling_mean(d, weights=weights)
    return x.map_batches(lambda x1: batches_i1_o1(x1.to_numpy().astype(float), roll_decay_exp_window, d, factor))


def ts_decay_linear(x: Expr, d: int = 30) -> Expr:
    """线性衰减移动平均

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 6, 5, 4, 5, 30],
    }).with_columns(
        out1=ts_decay_linear(pl.col('a'), 5),
        out2=ts_WMA(pl.col('a'), 5),
    )
    shape: (6, 3)
    ┌──────┬──────┬──────┐
    │ a    ┆ out1 ┆ out2 │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ f64  ┆ f64  │
    ╞══════╪══════╪══════╡
    │ null ┆ null ┆ null │
    │ 6    ┆ null ┆ null │
    │ 5    ┆ null ┆ null │
    │ 4    ┆ null ┆ null │
    │ 5    ┆ null ┆ null │
    │ 30   ┆ 13.2 ┆ 13.2 │
    └──────┴──────┴──────┘
    ```

    References
    ----------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#ts_decay_linearx-d-dense-false

    """
    # # weights not yet supported on array with null values
    # weights = arange(1, d + 1, eager=True)
    # # print(weights)
    # return x.rolling_mean(d, weights=weights)
    return x.map_batches(lambda x1: batches_i1_o1(x1.to_numpy().astype(float), roll_decay_linear, d))


def ts_delay(x: Expr, d: int = 1, fill_value=None) -> Expr:
    """时序数据移动 shift x

    Parameters
    ----------
    x
    d
        向前或向后的移动天数
    fill_value
        填充。可用None、常量或Expr

    """
    return x.shift(d, fill_value=fill_value)


def ts_delta(x: Expr, d: int = 1) -> Expr:
    """差分"""
    return x.diff(d)


def ts_fill_null(x: Expr) -> Expr:
    """用上一个非空值填充空值"""
    return x.forward_fill()


def ts_ir(x: Expr, d: int = 1) -> Expr:
    """时序滚动信息系数rolling information ratio"""
    return ts_mean(x, d) / ts_std_dev(x, d, 0)


def ts_kurtosis(x: Expr, d: int = 5) -> Expr:
    """kurtosis of x for the last d days

    时序滚动峰度

    Warnings
    --------
    等待polars官方出rolling_kurt

    """
    return x.map_batches(lambda a: roll_kurt(a, d))


def ts_l2_norm(x: Expr, d: int = 5) -> Expr:
    """Euclidean norm

    欧几里得范数"""
    return x.pow(2).rolling_sum(d).sqrt()


def ts_log_diff(x: Expr, d: int = 1) -> Expr:
    """对数差分。log(current value of input or x[t] ) - log(previous value of input or x[t-1]).

    """
    return x.log().diff(d)


def ts_max(x: Expr, d: int = 30) -> Expr:
    """时序滚动最大值"""
    return x.rolling_max(d)


def ts_max_diff(x: Expr, d: int = 30) -> Expr:
    """Returns x - ts_max(x, d)"""
    return x - ts_max(x, d)


def ts_mean(x: Expr, d: int = 5) -> Expr:
    """简单移动平均"""
    return x.rolling_mean(d)


def ts_median(x: Expr, d: int = 5) -> Expr:
    """时序滚动中位数"""
    return x.rolling_median(d)


def ts_min(x: Expr, d: int = 30) -> Expr:
    """时序滚动最小值"""
    return x.rolling_min(d)


def ts_min_diff(x: Expr, d: int = 30) -> Expr:
    """Returns x - ts_min(x, d)"""
    return x - ts_min(x, d)


def ts_min_max_cps(x: Expr, d: int, f: float = 2.0) -> Expr:
    """Returns (ts_min(x, d) + ts_max(x, d)) - f * x"""
    return (ts_min(x, d) + ts_max(x, d)) - f * x


def ts_min_max_diff(x: Expr, d: int, f: float = 0.5) -> Expr:
    """Returns x - f * (ts_min(x, d) + ts_max(x, d))"""
    return x - f * (ts_min(x, d) + ts_max(x, d))


def ts_moment(x: Expr, d: int, k: int = 0) -> Expr:
    """Returns K-th central moment of x for the past d days.

    滚动k阶中心距

    Parameters
    ----------
    x
    d
    k

    """
    return x.map_batches(lambda x1: batches_i1_o1(x1.to_numpy(), roll_moment, d, k))


def ts_partial_corr(x: Expr, y: Expr, z: Expr, d: int) -> Expr:
    """Returns partial correlation of x, y, z for the past d days.

    滚动偏相关
    """
    return struct([x, y, z]).map_batches(lambda xx: batches_i2_o1([xx.struct[i].to_numpy() for i in range(3)], roll_partial_corr, d))


def ts_percentage(x: Expr, d: int, percentage: float = 0.5) -> Expr:
    """Returns percentile value of x for the past d days.

    滚动百分位数

    Parameters
    ----------
    x
    d
    percentage

    """
    return x.rolling_quantile(percentage, window_size=d)


def ts_product(x: Expr, d: int = 5) -> Expr:
    """时序滚动乘"""
    return x.map_batches(lambda x1: batches_i1_o1(x1.to_numpy(), roll_prod, d))


def ts_rank(x: Expr, d: int = 5) -> Expr:
    """时序滚动排名

    Warnings
    --------
    等待polars官方出rolling_rank

    """
    return x.map_batches(lambda a: roll_rank(a, d, True))


def ts_returns(x: Expr, d: int = 1) -> Expr:
    """简单收益率"""
    return x.pct_change(d)


def ts_scale(x: Expr, d: int = 5) -> Expr:
    """Returns (x – ts_min(x, d)) / (ts_max(x, d) – ts_min(x, d)) + constant

    时序滚动缩放
    """
    a = ts_min(x, d)
    b = ts_max(x, d)
    # return (x - a) / (b - a + TA_EPSILON)
    return when(a != b).then((x - a) / (b - a)).otherwise(0)


def ts_skewness(x: Expr, d: int = 5, bias: bool = False) -> Expr:
    """Return skewness of x for the past d days

    时序滚动偏度

    Parameters
    ----------
    x
    d
    bias
        有偏

    Notes
    -----
    `bias=False`时与`pandas`结果一样

    """
    return x.rolling_skew(d, bias=bias)


def ts_std_dev(x: Expr, d: int = 5, ddof: int = 0) -> Expr:
    """时序滚动标准差

    Parameters
    ----------
    x
    d
    ddof
        自由度

    """
    return x.rolling_std(d, ddof=ddof)


def ts_sum(x: Expr, d: int = 30) -> Expr:
    """时序滚动求和"""
    return x.rolling_sum(d)


def ts_sum_split_by(x: Expr, by: Expr, d: int = 30, k: int = 10) -> Expr:
    """切割论求和。在d窗口范围内以by为依据进行从小到大排序。取最大的N个和最小的N个对应位置的x的和

    Parameters
    ----------
    x
    by
    d
        窗口大小
    k
        最大最小的k个

    Returns
    -------
    Expr
        * top_k
        * bottom_k

    Examples
    --------

    ```python
    df = pl.DataFrame({
        'a': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'b': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    }).with_columns(
        A=ts_sum_split_by(pl.col('a'), pl.col('b'), 8, 3)
    )

    shape: (10, 3)
    ┌─────┬─────┬───────────────┐
    │ a   ┆ b   ┆ A             │
    │ --- ┆ --- ┆ ---           │
    │ i64 ┆ i64 ┆ struct[2]     │
    ╞═════╪═════╪═══════════════╡
    │ 10  ┆ 10  ┆ {null,null}   │
    │ 20  ┆ 9   ┆ {null,null}   │
    │ 30  ┆ 8   ┆ {null,null}   │
    │ 40  ┆ 7   ┆ {null,null}   │
    │ 50  ┆ 6   ┆ {null,null}   │
    │ 60  ┆ 5   ┆ {null,null}   │
    │ 70  ┆ 4   ┆ {null,null}   │
    │ 80  ┆ 3   ┆ {210.0,60.0}  │
    │ 90  ┆ 2   ┆ {240.0,90.0}  │
    │ 100 ┆ 1   ┆ {270.0,120.0} │
    └─────┴─────┴───────────────┘
    ```

    """
    dtype = Struct([Field(f"column_{i}", Float64) for i in range(2)])
    return struct([x, by]).map_batches(lambda xx: batches_i2_o2([xx.struct[i].to_numpy() for i in range(2)], _sum_split_by, d, k), return_dtype=dtype)


def ts_triple_corr(x: Expr, y: Expr, z: Expr, d: int) -> Expr:
    """时序滚动三重相关系数 Returns triple correlation of x, y, z for the past d days.


    """
    return struct([x, y, z]).map_batches(lambda xx: batches_i2_o1([xx.struct[i].to_numpy() for i in range(3)], roll_triple_corr, d))


def ts_weighted_decay(x: Expr, k: float = 0.5) -> Expr:
    """加权衰减 Instead of replacing today’s value with yesterday’s as in ts_delay(x, 1),
    it assigns weighted average of today’s and yesterday’s values with weight on today’s value being k and yesterday’s being (1-k).

    Parameters
    ----------
    x
    k
        衰减系数

    """
    return x.rolling_sum(2, weights=[1 - k, k])


def ts_zscore(x: Expr, d: int = 5) -> Expr:
    """时序滚动zscore"""
    return (x - ts_mean(x, d)) / ts_std_dev(x, d, 0)


def ts_cum_prod_by(r: Expr, v: Expr) -> Expr:
    """带设置的累乘

    可用于市值累乘日收益率得到新市值的需求

    Parameters
    ----------
    r
        收益率。不能出现`null`, `null`需要提前用`1`代替

        `CLOSE/ts_delay(CLOSE, 1)`
    v
        市值。非空时分配指定市值资产

        * 如果非`null`，直接返回`v`
        * 如果`null`，返回`V[-1]*r`

    Returns
    -------
    Expr
        V。累乘后成新的市值

    Examples
    --------

    ```python
    df = pl.DataFrame({
        'r': [1, 2, 3, 4, 5, 6],
        'v': [None, None, 6, None, None, 12],

    }).with_columns(
        V=ts_cum_prod_by(pl.col('r'), pl.col('v'))
    )

    shape: (6, 3)
    ┌─────┬──────┬───────┐
    │ r   ┆ v    ┆ V     │
    │ --- ┆ ---  ┆ ---   │
    │ i64 ┆ i64  ┆ f64   │
    ╞═════╪══════╪═══════╡
    │ 1   ┆ null ┆ null  │
    │ 2   ┆ null ┆ null  │
    │ 3   ┆ 6    ┆ 6.0   │
    │ 4   ┆ null ┆ 24.0  │
    │ 5   ┆ null ┆ 120.0 │
    │ 6   ┆ 12   ┆ 12.0  │
    └─────┴──────┴───────┘
    ```


    """
    return struct([r, v]).map_batches(lambda xx: batches_i2_o1([xx.struct[i].to_numpy().astype(float) for i in range(2)], _cum_prod_by))


def ts_cum_sum_by(r: Expr, v: Expr) -> Expr:
    """带设置的累加

    可用于市值累加日收益得到新市值的需求

    Parameters
    ----------
    r
        收益。不能出现`null`, `null`需要提前用`0`代替

        `CLOSE-ts_delay(CLOSE, 1)`
    v
        市值。非空时分配指定市值资产

        * 如果非`null`，直接返回`v`
        * 如果`null`，返回`V[-1]+r`

    Examples
    --------

    ```python
    df = pl.DataFrame({
        'r': [1, 2, 3, 4, 5, 6],
        'v': [None, None, 6, None, None, 12],

    }).with_columns(
        V=ts_cum_sum_by(pl.col('r'), pl.col('v'))
    )

    shape: (6, 3)
    ┌─────┬──────┬──────┐
    │ r   ┆ v    ┆ V    │
    │ --- ┆ ---  ┆ ---  │
    │ i64 ┆ i64  ┆ f64  │
    ╞═════╪══════╪══════╡
    │ 1   ┆ null ┆ null │
    │ 2   ┆ null ┆ null │
    │ 3   ┆ 6    ┆ 6.0  │
    │ 4   ┆ null ┆ 10.0 │
    │ 5   ┆ null ┆ 15.0 │
    │ 6   ┆ 12   ┆ 12.0 │
    └─────┴──────┴──────┘

    ```

    """
    return struct([r, v]).map_batches(lambda xx: batches_i2_o1([xx.struct[i].to_numpy().astype(float) for i in range(2)], _cum_sum_by))


def ts_regression_resid(y: Expr, x: Expr, d: int) -> Expr:
    """时序滚动回归取残差"""
    return pls.compute_rolling_least_squares(y, x, mode='residuals', add_intercept=True, rolling_kwargs=RollingKwargs(window_size=d, min_periods=d))


def ts_regression_pred(y: Expr, x: Expr, d: int) -> Expr:
    """时序滚动回归取y的预测值
    """
    return pls.compute_rolling_least_squares(y, x, mode='predictions', add_intercept=True, rolling_kwargs=RollingKwargs(window_size=d, min_periods=d))


def ts_regression_intercept(y: Expr, x: Expr, d: int) -> Expr:
    """时序滚动回归取截距
    """
    return pls.compute_rolling_least_squares(y, x, mode='coefficients', add_intercept=True, rolling_kwargs=RollingKwargs(window_size=d, min_periods=d)).struct[1]


def ts_regression_slope(y: Expr, x: Expr, d: int) -> Expr:
    """时序滚动回归取斜率"""
    return pls.compute_rolling_least_squares(y, x, mode='coefficients', add_intercept=True, rolling_kwargs=RollingKwargs(window_size=d, min_periods=d)).struct[0]


def ts_resid(y: Expr, *more_x: Expr, d: int = 30) -> Expr:
    """多元时序滚动回归取残差

    Parameters
    ----------
    y
    *more_x
        多个x
    d

    """
    return pls.compute_rolling_least_squares(y, *more_x, mode='residuals', rolling_kwargs=RollingKwargs(window_size=d, min_periods=d))


def ts_pred(y: Expr, *more_x: Expr, d: int = 30) -> Expr:
    """多元时序滚动回归预测

    Parameters
    ----------
    y
    *more_x
        多个x
    d

    """
    return pls.compute_rolling_least_squares(y, *more_x, mode='predictions', rolling_kwargs=RollingKwargs(window_size=d, min_periods=d))


def ts_weighted_mean(x: Expr, w: Expr, d: int) -> Expr:
    """时序滚动加权平均"""
    return (x * w).rolling_sum(d) / w.rolling_sum(d)


def ts_weighted_sum(x: Expr, w: Expr, d: int) -> Expr:
    """时序滚动加权求和"""
    return (x * w).rolling_sum(d)


def ts_signals_to_size(long_entry: Expr, long_exit: Expr,
                       short_entry: Expr, short_exit: Expr,
                       accumulate: bool = False,
                       action: bool = False) -> Expr:
    """多空信号转持仓。参考于`vectorbt`

    Parameters
    ----------
    long_entry
        多头入场
    long_exit
        多头出场
    short_entry
        空头入场
    short_exit
        空头出场
    accumulate
        遇到重复信号时是否累计
    action
        返回持仓状态还是下单操作

    """
    return struct([long_entry, long_exit, short_entry, short_exit]).map_batches(
        lambda xx: batches_i2_o1([xx.struct[i].to_numpy().astype(float) for i in range(4)],
                                 _signals_to_size, accumulate, action))
