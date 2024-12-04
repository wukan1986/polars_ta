from polars import Expr, when, Boolean, Int32


def cut(x: Expr, b: float, *more_bins) -> Expr:
    """分箱

    Parameters
    ----------
    x
    b
    *more_bins

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 1, 1, 1, 2, 2, 3, 10],
    }).with_columns(
        out1=cut(pl.col('a'), 2, 5, 20),
    )
    shape: (8, 2)
    ┌──────┬──────┐
    │ a    ┆ out1 │
    │ ---  ┆ ---  │
    │ i64  ┆ u32  │
    ╞══════╪══════╡
    │ null ┆ null │
    │ 1    ┆ 0    │
    │ 1    ┆ 0    │
    │ 1    ┆ 0    │
    │ 2    ┆ 0    │
    │ 2    ┆ 0    │
    │ 3    ┆ 1    │
    │ 10   ┆ 2    │
    └──────┴──────┘

    ```

    """
    return x.cut([b, *more_bins]).to_physical()


def clamp(x: Expr, lower: float = 0, upper: float = 0, inverse: bool = False, mask: float = None) -> Expr:
    """Limits input value between lower and upper bound in inverse = false mode (which is default). Alternatively, when inverse = true, values between bounds are replaced with mask, while values outside bounds are left as is.

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 1, 2, 3, 4, 5, 6],
    }).with_columns(
        out1=clamp(pl.col('a'), 2, 5, False),
        out2=clamp(pl.col('a'), 2, 5, True),
    )
    shape: (7, 3)
    ┌──────┬──────┬──────┐
    │ a    ┆ out1 ┆ out2 │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  │
    ╞══════╪══════╪══════╡
    │ null ┆ null ┆ null │
    │ 1    ┆ 2    ┆ 1    │
    │ 2    ┆ 2    ┆ null │
    │ 3    ┆ 3    ┆ null │
    │ 4    ┆ 4    ┆ null │
    │ 5    ┆ 5    ┆ null │
    │ 6    ┆ 5    ┆ 6    │
    └──────┴──────┴──────┘

    ```

    References
    ----------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#clampx-lower-0-upper-0-inverse-false-mask

    """
    if inverse:
        cond = (x >= lower) & (x <= upper)
        return when(~cond).then(x).otherwise(mask)
    else:
        return x.clip(lower, upper)


# def filter_(x: Expr, h: str = "1, 2, 3, 4", t: str = "0.5") -> Expr:
#     """Used to filter the value and allows to create filters like linear or exponential decay."""
#     raise


def keep(x: Expr, f: float, period: int = 5) -> Expr:
    """This operator outputs value x when f changes and continues to do that for “period” days after f stopped changing. After “period” days since last change of f, NaN is output."""
    raise


def left_tail(x: Expr, maximum: float = 0) -> Expr:
    """NaN everything greater than maximum, maximum should be constant.

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 1, 2, 3, 4, 5],
    }).with_columns(
        out=left_tail(pl.col('a'), 3),
    )
    shape: (6, 2)
    ┌──────┬──────┐
    │ a    ┆ out  │
    │ ---  ┆ ---  │
    │ i64  ┆ i64  │
    ╞══════╪══════╡
    │ null ┆ null │
    │ 1    ┆ 1    │
    │ 2    ┆ 2    │
    │ 3    ┆ 3    │
    │ 4    ┆ null │
    │ 5    ┆ null │
    └──────┴──────┘
    ```

    See Also
    --------
    tail

    References
    ----------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#left_tail

    """
    return when(x <= maximum).then(x).otherwise(None)


# def pasteurize(x: Expr) -> Expr:
#     """Set to NaN if x is INF or if the underlying instrument is not in the Alpha universe"""
#     # TODO: 不在票池中的的功能无法表示
#     # TODO: 与purify好像没啥区别
#     return when(x.is_finite()).then(x).otherwise(None)


def purify(x: Expr) -> Expr:
    """Clear infinities (+inf, -inf) by replacing with null.

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 1., 2., float('nan'), float('inf'), float('-inf')],
    }).with_columns(
        out=purify(pl.col('a')),
    )
    shape: (6, 2)
    ┌──────┬──────┐
    │ a    ┆ out  │
    │ ---  ┆ ---  │
    │ f64  ┆ f64  │
    ╞══════╪══════╡
    │ null ┆ null │
    │ 1.0  ┆ 1.0  │
    │ 2.0  ┆ 2.0  │
    │ NaN  ┆ null │
    │ inf  ┆ null │
    │ -inf ┆ null │
    └──────┴──────┘
    ```

    """
    return when(x.is_finite()).then(x).otherwise(None)


def fill_nan(x: Expr) -> Expr:
    """填充`nan`为`null`"""
    return x.fill_nan(None)


def fill_null(x: Expr, value=0) -> Expr:
    """填充`null`为`value`"""
    return x.fill_null(value)


def right_tail(x: Expr, minimum: float = 0) -> Expr:
    """NaN everything less than minimum, minimum should be constant.

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 1, 2, 3, 4, 5],
    }).with_columns(
        out=right_tail(pl.col('a'), 3),
    )
    shape: (6, 2)
    ┌──────┬──────┐
    │ a    ┆ out  │
    │ ---  ┆ ---  │
    │ i64  ┆ i64  │
    ╞══════╪══════╡
    │ null ┆ null │
    │ 1    ┆ null │
    │ 2    ┆ null │
    │ 3    ┆ 3    │
    │ 4    ┆ 4    │
    │ 5    ┆ 5    │
    └──────┴──────┘
    ```

    References
    ----------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#right_tail

    """
    return when(x >= minimum).then(x).otherwise(None)


def sigmoid(x: Expr) -> Expr:
    """Returns 1 / (1 + exp(-x))"""
    return 1 / (1 + (-x).exp())


def tail(x: Expr, lower: float = 0, upper: float = 0, newval: float = 0) -> Expr:
    """If (x > lower AND x < upper) return newval, else return x. Lower, upper, newval should be constants.

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 1, 2, 3, 4, 5, 6],
    }).with_columns(
        out=tail(pl.col('a'), 2, 5, 1),
    )
    shape: (7, 2)
    ┌──────┬──────┐
    │ a    ┆ out  │
    │ ---  ┆ ---  │
    │ i64  ┆ i64  │
    ╞══════╪══════╡
    │ null ┆ null │
    │ 1    ┆ 1    │
    │ 2    ┆ 2    │
    │ 3    ┆ 1    │
    │ 4    ┆ 1    │
    │ 5    ┆ 5    │
    │ 6    ┆ 6    │
    └──────┴──────┘

    ```

    See Also
    --------
    clamp

    References
    ----------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#tail

    """

    cond = (x > lower) & (x < upper)
    return when(~cond | x.is_null()).then(x).otherwise(newval)


def int_(a: Expr) -> Expr:
    """convert bool to int"""
    return a.cast(Int32)


def bool_(a: Expr) -> Expr:
    """convert int to bool"""
    return a.cast(Boolean)
