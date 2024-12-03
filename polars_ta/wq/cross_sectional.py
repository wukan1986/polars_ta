import polars_ols as pls
from polars import Expr, when
from polars_ols import OLSKwargs

# In the original version, the function names are not prefixed with `cs_`,
# here we add it to prevent confusion
# 原版函数名都没有加`cs_`, 这里统一加一防止混淆


_ols_kwargs = OLSKwargs(null_policy='drop', solve_method='svd')


def cs_one_side(x: Expr, is_long: bool = True) -> Expr:
    """Shifts all instruments up or down so that the Alpha becomes long-only or short-only
(if side = short), respectively."""
    # TODO: 这里不确定，需再研究
    # [-1, 0, 1]+1=[0, 1, 2]
    # max([-1, 0, 1], 0)=[0,0,1]
    raise


def cs_rank(x: Expr, pct: bool = True) -> Expr:
    """排名。Ranks the input among all the instruments and returns an equally distributed number between 0.0 and 1.0. For precise sort, use the rate as 0.

    Parameters
    ----------
    x
    pct
        * True: 排名百分比。范围：(0,1]
        * False: 排名。范围：[1,+inf)

    Examples
    --------

    ```python
    df = pl.DataFrame({
        'a': [None, 1, 1, 1, 2, 2, 3, 10],
    }).with_columns(
        out1=cs_rank(pl.col('a'), True),
        out2=cs_rank(pl.col('a'), False),
    )

    shape: (8, 3)
    ┌──────┬──────────┬──────┐
    │ a    ┆ out1     ┆ out2 │
    │ ---  ┆ ---      ┆ ---  │
    │ i64  ┆ f64      ┆ u32  │
    ╞══════╪══════════╪══════╡
    │ null ┆ null     ┆ null │
    │ 1    ┆ 0.142857 ┆ 1    │
    │ 1    ┆ 0.142857 ┆ 1    │
    │ 1    ┆ 0.142857 ┆ 1    │
    │ 2    ┆ 0.571429 ┆ 4    │
    │ 2    ┆ 0.571429 ┆ 4    │
    │ 3    ┆ 0.857143 ┆ 6    │
    │ 10   ┆ 1.0      ┆ 7    │
    └──────┴──────────┴──────┘
    ```

    """
    if pct:
        return x.rank(method='min') / x.count()
    else:
        return x.rank(method='min')


def cs_scale(x: Expr, scale_: float = 1, long_scale: float = 1, short_scale: float = 1) -> Expr:
    """Scales input to booksize. We can also scale the long positions and short positions to separate scales by mentioning additional parameters to the operator."""
    if long_scale != 1 or short_scale != 1:
        L = x.clip(lower_bound=0)  # 全正数
        S = x.clip(upper_bound=0)  # 全负数，和还是负数
        return L / L.sum() * long_scale - S / S.sum() * short_scale
    else:
        return x / x.abs().sum() * scale_


def cs_truncate(x: Expr, max_percent: float = 0.01) -> Expr:
    """Operator truncates all values of x to maxPercent. Here, maxPercent is in decimal notation

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [3, 7, 20, 6],
    }).with_columns(
        out=cs_truncate(pl.col('a'), 0.5),
    )
    shape: (4, 2)
    ┌─────┬─────┐
    │ a   ┆ out │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 3   ┆ 3   │
    │ 7   ┆ 7   │
    │ 20  ┆ 18  │
    │ 6   ┆ 6   │
    └─────┴─────┘
    ```

    Reference
    ---------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#truncatexmaxpercent001

    """
    return x.clip(upper_bound=x.sum() * max_percent)


def cs_fill_except_all_null(x: Expr, value=0) -> Expr:
    """全为`null`时，保持`null`，反之`null`填充为`value`

    Examples
    --------

    ```python
    df = pl.DataFrame({
        'a': [1, 2, None, 4, None],
        'b': [None, None, None, None, None],
    }).with_columns(
        A=cs_fill_except_all_null(pl.col('a')),
        B=cs_fill_except_all_null(pl.col('b')),
    )

    shape: (5, 4)
    ┌──────┬──────┬─────┬──────┐
    │ a    ┆ b    ┆ A   ┆ B    │
    │ ---  ┆ ---  ┆ --- ┆ ---  │
    │ i64  ┆ null ┆ i64 ┆ i32  │
    ╞══════╪══════╪═════╪══════╡
    │ 1    ┆ null ┆ 1   ┆ null │
    │ 2    ┆ null ┆ 2   ┆ null │
    │ null ┆ null ┆ 0   ┆ null │
    │ 4    ┆ null ┆ 4   ┆ null │
    │ null ┆ null ┆ 0   ┆ null │
    └──────┴──────┴─────┴──────┘
    ```

    Notes
    -----
    在权重矩阵中使用时。一定要保证所有股票都在，停牌不能被过滤了

    """
    return when(x.is_not_null().sum() == 0).then(x).otherwise(x.fill_null(value))


def cs_regression_neut(y: Expr, x: Expr) -> Expr:
    """一元回归残差"""
    return pls.compute_least_squares(y, x, add_intercept=True, mode='residuals', ols_kwargs=_ols_kwargs)


def cs_regression_proj(y: Expr, x: Expr) -> Expr:
    """一元回归预测"""
    return pls.compute_least_squares(y, x, add_intercept=True, mode='predictions', ols_kwargs=_ols_kwargs)


def cs_qcut(x: Expr, q: int = 10) -> Expr:
    """等频分箱 Convert float values into indexes for user-specified buckets. Bucket is useful for creating group values, which can be passed to group operators as input.

    Parameters
    ----------
    x
    q
        按频率分成`q`份

    Examples
    --------

    ```python
    df = pl.DataFrame({
        'a': [None, 1, 1, 1, 2, 2, 3, 10],
    }).with_columns(
        out1=cs_qcut(pl.col('a'), 10),
        out2=pl.col('a').map_batches(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')),
    )
    shape: (8, 3)
    ┌──────┬──────┬──────┐
    │ a    ┆ out1 ┆ out2 │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ u32  ┆ f64  │
    ╞══════╪══════╪══════╡
    │ null ┆ null ┆ NaN  │
    │ 1    ┆ 0    ┆ 0.0  │
    │ 1    ┆ 0    ┆ 0.0  │
    │ 1    ┆ 0    ┆ 0.0  │
    │ 2    ┆ 3    ┆ 1.0  │
    │ 2    ┆ 3    ┆ 1.0  │
    │ 3    ┆ 7    ┆ 4.0  │
    │ 10   ┆ 8    ┆ 5.0  │
    └──────┴──────┴──────┘

    ```

    Warnings
    --------
    目前与`pd.qcut`结果不同，等官方改进

    """
    return x.qcut(q, allow_duplicates=True).to_physical()
