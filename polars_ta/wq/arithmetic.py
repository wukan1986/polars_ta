import numpy as np
from polars import Expr, Series, fold, any_horizontal, Float64, Int64
from polars import arctan2 as _arctan2
from polars import max_horizontal, sum_horizontal, min_horizontal, mean_horizontal


def abs_(x: Expr) -> Expr:
    """求绝对值

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
        'b': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=abs_(pl.col('a')),
        out2=abs_(-1),
    )
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ out1 ┆ out2 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  ┆ i32  │
    ╞══════╪══════╪══════╪══════╡
    │ null ┆ null ┆ null ┆ 1    │
    │ -1   ┆ -1   ┆ 1    ┆ 1    │
    │ 0    ┆ 0    ┆ 0    ┆ 1    │
    │ 1    ┆ 1    ┆ 1    ┆ 1    │
    │ 2    ┆ 2    ┆ 2    ┆ 1    │
    └──────┴──────┴──────┴──────┘
    ```

    """
    if isinstance(x, (Expr, Series)):
        return x.abs()
    else:
        return np.abs(x)


def add(a: Expr, b: Expr, *args) -> Expr:
    """水平多列相加

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 2, 3, 4, None],
        'b': [5, None, 3, 2, None],
        'c': [1, 1, None, 1, None],
    }).with_columns(
        out=add(pl.col('a'), pl.col('b'), pl.col('c'))
    )
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ c    ┆ out  │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  ┆ i64  │
    ╞══════╪══════╪══════╪══════╡
    │ null ┆ 5    ┆ 1    ┆ 6    │
    │ 2    ┆ null ┆ 1    ┆ 3    │
    │ 3    ┆ 3    ┆ null ┆ 6    │
    │ 4    ┆ 2    ┆ 1    ┆ 7    │
    │ null ┆ null ┆ null ┆ null │
    └──────┴──────┴──────┴──────┘

    ```

    Notes
    -----
    全`null`时返回`null`

    """
    # # 全null时返回0
    # return sum_horizontal(a, b, *args)

    _args = [a, b] + list(args)
    return fold(acc=any_horizontal(_args) - 1, function=lambda acc, x: acc + x.fill_null(0), exprs=_args)


def arc_cos(x: Expr) -> Expr:
    """反余弦"""
    return x.arccos()


def arc_sin(x: Expr) -> Expr:
    """反正弦"""
    return x.arcsin()


def arc_tan(x: Expr) -> Expr:
    """反正切"""
    return x.arctan()


def arc_tan2(y: Expr, x: Expr) -> Expr:
    """反正切二值函数"""
    return _arctan2(y, x)


def cbrt(x: Expr) -> Expr:
    """立方根"""
    return x.cbrt()


def ceiling(x: Expr) -> Expr:
    """向上取整"""
    return x.ceil()


def cos(x: Expr) -> Expr:
    """余弦"""
    return x.cos()


def cosh(x: Expr) -> Expr:
    """双曲余弦"""
    return x.cosh()


def cot(x: Expr) -> Expr:
    """余切"""
    return x.cot()


def cube(x: Expr) -> Expr:
    """立方"""
    return x.pow(3)


def degrees(x: Expr) -> Expr:
    """弧度转角度"""
    return x.degrees()


def _densify(x: Expr) -> Expr:
    raise


def div(x: Expr, y: Expr) -> Expr:
    """x除以y的整数部分

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1.5, 0., 1.5, 2.5],
        'b': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=div(pl.col('a'), 0),
        out2=div(pl.col('a'), 1),
        out3=div(pl.col('a'), pl.col('b')),
    )
    shape: (5, 5)
    ┌──────┬──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ out1 ┆ out2 ┆ out3 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ f64  ┆ i64  ┆ i64  ┆ i64  ┆ i64  │
    ╞══════╪══════╪══════╪══════╪══════╡
    │ null ┆ null ┆ null ┆ null ┆ null │
    │ -1.5 ┆ -1   ┆ null ┆ -2   ┆ 1    │
    │ 0.0  ┆ 0    ┆ null ┆ 0    ┆ null │
    │ 1.5  ┆ 1    ┆ null ┆ 1    ┆ 1    │
    │ 2.5  ┆ 2    ┆ null ┆ 2    ┆ 1    │
    └──────┴──────┴──────┴──────┴──────┘
    ```

    """
    return x.floordiv(y).cast(Int64, strict=False)


def divide(x: Expr, y: Expr) -> Expr:
    """除法

    x/y

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
        'b': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=divide(pl.col('a'), 0),
        out2=divide(pl.col('a'), 1),
        out3=divide(pl.col('a'), pl.col('b')),
    )
    shape: (5, 5)
    ┌──────┬──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ out1 ┆ out2 ┆ out3 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ f64  ┆ f64  ┆ f64  │
    ╞══════╪══════╪══════╪══════╪══════╡
    │ null ┆ null ┆ null ┆ null ┆ null │
    │ -1   ┆ -1   ┆ -inf ┆ -1.0 ┆ 1.0  │
    │ 0    ┆ 0    ┆ NaN  ┆ 0.0  ┆ NaN  │
    │ 1    ┆ 1    ┆ inf  ┆ 1.0  ┆ 1.0  │
    │ 2    ┆ 2    ┆ inf  ┆ 2.0  ┆ 1.0  │
    └──────┴──────┴──────┴──────┴──────┘
    ```

    """
    return x / y


def exp(x: Expr) -> Expr:
    """自然指数函数

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=expm1(pl.col('a')),
    )
    shape: (5, 2)
    ┌──────┬──────────┐
    │ a    ┆ out1     │
    │ ---  ┆ ---      │
    │ i64  ┆ f64      │
    ╞══════╪══════════╡
    │ null ┆ null     │
    │ -1   ┆ 0.367879 │
    │ 0    ┆ 1.0      │
    │ 1    ┆ 2.718282 │
    │ 2    ┆ 7.389056 │
    └──────┴──────────┘
    ```

    """
    return x.exp()


def expm1(x: Expr) -> Expr:
    """对数收益率 转 简单收益率

    convert log return to simple return

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=expm1(pl.col('a')),
    )
    shape: (5, 2)
    ┌──────┬───────────┐
    │ a    ┆ out1      │
    │ ---  ┆ ---       │
    │ i64  ┆ f64       │
    ╞══════╪═══════════╡
    │ null ┆ null      │
    │ -1   ┆ -0.632121 │
    │ 0    ┆ 0.0       │
    │ 1    ┆ 1.718282  │
    │ 2    ┆ 6.389056  │
    └──────┴───────────┘
    ```

    """
    return x.exp() - 1


def floor(x: Expr) -> Expr:
    """向下取整"""
    return x.floor()


def fraction(x: Expr) -> Expr:
    """小数部分

    This operator removes the whole number part and returns the remaining fraction part with sign.

    Examples
    --------

    ```python
    df = pl.DataFrame({
        'a': [-2.5, -1.2, -1., None, 2., 3.2],
    }).with_columns(
        out=fraction(pl.col('a'))
    )

    shape: (6, 2)
    ┌──────┬──────┐
    │ a    ┆ out  │
    │ ---  ┆ ---  │
    │ f64  ┆ f64  │
    ╞══════╪══════╡
    │ -2.5 ┆ -0.5 │
    │ -1.2 ┆ -0.2 │
    │ -1.0 ┆ -0.0 │
    │ null ┆ null │
    │ 2.0  ┆ 0.0  │
    │ 3.2  ┆ 0.2  │
    └──────┴──────┘
    ```

    Notes
    -----
    按小学时的定义，负数`-1.2`的整数部分是`-2`,小数部分是`0.8`，而这有所不同

    References
    ----------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#fractionx

    """
    return x.sign() * (x.abs() % 1)


def inverse(x: Expr) -> Expr:
    """倒数

    1/x

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=inverse(pl.col('a')),
    )
    shape: (5, 2)
    ┌──────┬──────┐
    │ a    ┆ out1 │
    │ ---  ┆ ---  │
    │ i64  ┆ f64  │
    ╞══════╪══════╡
    │ null ┆ null │
    │ -1   ┆ -1.0 │
    │ 0    ┆ inf  │
    │ 1    ┆ 1.0  │
    │ 2    ┆ 0.5  │
    └──────┴──────┘
    ```

    """
    return 1 / x


def log(x: Expr) -> Expr:
    """以e为底的对数

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=log(pl.col('a')),
    )
    shape: (5, 2)
    ┌──────┬──────────┐
    │ a    ┆ out1     │
    │ ---  ┆ ---      │
    │ i64  ┆ f64      │
    ╞══════╪══════════╡
    │ null ┆ null     │
    │ -1   ┆ NaN      │
    │ 0    ┆ -inf     │
    │ 1    ┆ 0.0      │
    │ 2    ┆ 0.693147 │
    └──────┴──────────┘
    ```

    """
    if isinstance(x, (Expr, Series)):
        return x.log()
    else:
        return np.log(x)


def log10(x: Expr) -> Expr:
    """以10为底的对数

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=log10(pl.col('a')),
    )
    shape: (5, 2)
    ┌──────┬─────────┐
    │ a    ┆ out1    │
    │ ---  ┆ ---     │
    │ i64  ┆ f64     │
    ╞══════╪═════════╡
    │ null ┆ null    │
    │ -1   ┆ NaN     │
    │ 0    ┆ -inf    │
    │ 1    ┆ 0.0     │
    │ 2    ┆ 0.30103 │
    └──────┴─────────┘
    ```

    """
    return x.log10()


def log1p(x: Expr) -> Expr:
    """简单收益率 转 对数收益率

    convert simple return to log return

    log(x+1)

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=log1p(pl.col('a')),
    )
    shape: (5, 2)
    ┌──────┬──────────┐
    │ a    ┆ out1     │
    │ ---  ┆ ---      │
    │ i64  ┆ f64      │
    ╞══════╪══════════╡
    │ null ┆ null     │
    │ -1   ┆ -inf     │
    │ 0    ┆ 0.0      │
    │ 1    ┆ 0.693147 │
    │ 2    ┆ 1.098612 │
    └──────┴──────────┘
    ```

    """
    return x.log1p()


def log2(x: Expr) -> Expr:
    """以2为底的对数

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=log2(pl.col('a')),
    )
    shape: (5, 2)
    ┌──────┬──────┐
    │ a    ┆ out1 │
    │ ---  ┆ ---  │
    │ i64  ┆ f64  │
    ╞══════╪══════╡
    │ null ┆ null │
    │ -1   ┆ NaN  │
    │ 0    ┆ -inf │
    │ 1    ┆ 0.0  │
    │ 2    ┆ 1.0  │
    └──────┴──────┘
    ```

    """
    return x.log(2)


def max_(a: Expr, b: Expr, *args) -> Expr:
    """水平多列求最大值

    Maximum value of all inputs. At least 2 inputs are required."""
    return max_horizontal(a, b, *args)


def mean(a: Expr, b: Expr, *args) -> Expr:
    """水平多列求均值

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
        'b': [None, -1, 0, 0, 2],
    }).with_columns(
        out2=mean(pl.col('a'), 2),
        out3=mean(pl.col('a'), pl.col('b')),
    )
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ out2 ┆ out3 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ f64  ┆ f64  │
    ╞══════╪══════╪══════╪══════╡
    │ null ┆ null ┆ 2.0  ┆ null │
    │ -1   ┆ -1   ┆ 0.5  ┆ -1.0 │
    │ 0    ┆ 0    ┆ 1.0  ┆ 0.0  │
    │ 1    ┆ 0    ┆ 1.5  ┆ 0.5  │
    │ 2    ┆ 2    ┆ 2.0  ┆ 2.0  │
    └──────┴──────┴──────┴──────┘
    ```

    """
    return mean_horizontal(a, b, *args)


def min_(a: Expr, b: Expr, *args) -> Expr:
    """水平多列求最小值

    Maximum value of all inputs. At least 2 inputs are required."""
    return min_horizontal(a, b, *args)


def mod(x: Expr, y: Expr) -> Expr:
    """求余

    x%y

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
        'b': [None, -1, 0, 0, 2],
    }).with_columns(
        out2=mod(pl.col('a'), 2),
        out3=mod(pl.col('a'), pl.col('b')),
    )
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ out2 ┆ out3 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  ┆ i64  │
    ╞══════╪══════╪══════╪══════╡
    │ null ┆ null ┆ null ┆ null │
    │ -1   ┆ -1   ┆ 1    ┆ 0    │
    │ 0    ┆ 0    ┆ 0    ┆ null │
    │ 1    ┆ 0    ┆ 1    ┆ null │
    │ 2    ┆ 2    ┆ 0    ┆ 0    │
    └──────┴──────┴──────┴──────┘
    ```

    """
    return x % y


def multiply(a: Expr, b: Expr, *args) -> Expr:
    """水平多列相乘

    Multiply all inputs. At least 2 inputs are required.

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 2, 3, 4, None],
        'b': [5, None, 3, 2, None],
        'c': [1, 1, None, 1, None],
    }).with_columns(
        out=multiply(pl.col('a'), pl.col('b'), pl.col('c'))
    )

    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ c    ┆ out  │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  ┆ i64  │
    ╞══════╪══════╪══════╪══════╡
    │ null ┆ 5    ┆ 1    ┆ 5    │
    │ 2    ┆ null ┆ 1    ┆ 2    │
    │ 3    ┆ 3    ┆ null ┆ 9    │
    │ 4    ┆ 2    ┆ 1    ┆ 8    │
    │ null ┆ null ┆ null ┆ null │
    └──────┴──────┴──────┴──────┘

    ```

    Notes
    -----
    全`null`时返回`null`

    """
    _args = [a, b] + list(args)

    # # 全null返回1
    # return fold(acc=1, function=lambda acc, x: acc * x.fill_null(1), exprs=_args)
    return fold(acc=any_horizontal(_args), function=lambda acc, x: acc * x.fill_null(1), exprs=_args)


def power(x: Expr, y: Expr) -> Expr:
    """乘幂

    x ** y

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
        'b': [None, -1, 0, 1, 2],
    }).with_columns(
        out2=power(pl.col('a'), 1),
        out3=power(pl.col('a'), pl.col('b')),
    )
    shape: (5, 4)
    ┌──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ out2 ┆ out3 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  ┆ f64  │
    ╞══════╪══════╪══════╪══════╡
    │ null ┆ null ┆ null ┆ null │
    │ -1   ┆ -1   ┆ -1   ┆ -1.0 │
    │ 0    ┆ 0    ┆ 0    ┆ 1.0  │
    │ 1    ┆ 1    ┆ 1    ┆ 1.0  │
    │ 2    ┆ 2    ┆ 2    ┆ 4.0  │
    └──────┴──────┴──────┴──────┘
    ```

    """
    if isinstance(y, (int, float)):
        return x.pow(y)

    return x.pow(y.cast(Float64))


def radians(x: Expr) -> Expr:
    """角度转弧度"""
    return x.radians()


def reverse(x: Expr) -> Expr:
    """求相反数"""
    return -x


def round_(x: Expr, decimals: int = 0) -> Expr:
    """四舍五入

    Round input to closest integer.

    Parameters
    ----------
    x
    decimals
        Number of decimals to round to.

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 3.5, 4.5, -3.5, -4.5],
    }).with_columns(
        out1=round_(pl.col('a'), 0),
        out2=pl.col('a').map_elements(lambda x: round(x, 0), return_dtype=pl.Float64),
    )
    shape: (5, 3)
    ┌──────┬──────┬──────┐
    │ a    ┆ out1 ┆ out2 │
    │ ---  ┆ ---  ┆ ---  │
    │ f64  ┆ f64  ┆ f64  │
    ╞══════╪══════╪══════╡
    │ null ┆ null ┆ null │
    │ 3.5  ┆ 4.0  ┆ 4.0  │
    │ 4.5  ┆ 5.0  ┆ 4.0  │
    │ -3.5 ┆ -4.0 ┆ -4.0 │
    │ -4.5 ┆ -5.0 ┆ -4.0 │
    └──────┴──────┴──────┘
    ```

    Notes
    -----
    四舍五入，不是四舍六入五取偶（银行家舍入）

    """
    return x.round(decimals)


def round_down(x: Expr, f: int = 1) -> Expr:
    """小于输入的f的最大倍数

    Round input to greatest multiple of f less than input

    Parameters
    ----------
    x
    f

    Examples
    --------

    ```python
    df = pl.DataFrame({
        'a': [None, 3.5, 4.5, -3.5, -4.5],
    }).with_columns(
        out=round_down(pl.col('a'), 2),
    )
    shape: (5, 2)
    ┌──────┬──────┐
    │ a    ┆ out  │
    │ ---  ┆ ---  │
    │ f64  ┆ f64  │
    ╞══════╪══════╡
    │ null ┆ null │
    │ 3.5  ┆ 2.0  │
    │ 4.5  ┆ 4.0  │
    │ -3.5 ┆ -4.0 │
    │ -4.5 ┆ -6.0 │
    └──────┴──────┘
    ```

    """
    if f == 1:
        return x // 1
    else:
        return x // f * f


def s_log_1p(x: Expr) -> Expr:
    """sign(x) * log10(1 + abs(x))

    一种结合符号函数和对数变换的复合函数，常用于‌保留数据符号的同时压缩数值范围‌

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, 9, -9, 99, -99],
    }).with_columns(
        out1=s_log_1p(pl.col('a')),
    )
    shape: (5, 2)
    ┌──────┬──────┐
    │ a    ┆ out1 │
    │ ---  ┆ ---  │
    │ i64  ┆ f64  │
    ╞══════╪══════╡
    │ null ┆ null │
    │ 9    ┆ 1.0  │
    │ -9   ┆ -1.0 │
    │ 99   ┆ 2.0  │
    │ -99  ┆ -2.0 │
    └──────┴──────┘

    ```

    Notes
    -----
    从`wq`示例可以看出，log的底数是10，而不是e

    References
    ----------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#s_log_1px

    """
    return (x.abs() + 1).log10() * x.sign()


def sign(x: Expr) -> Expr:
    """符号函数"""
    if isinstance(x, (Expr, Series)):
        return x.sign()
    else:
        return np.sign(x)


def signed_power(x: Expr, y: Expr) -> Expr:
    """x的y次幂，符号保留

    x raised to the power of y such that final result preserves sign of x.

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
        'b': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=signed_power(pl.col('a'), 0),
        out2=signed_power(pl.col('a'), 1),
        out3=signed_power(pl.col('a'), 2),
        out4=signed_power(pl.col('a'), pl.col('b')),
    )

    shape: (5, 6)
    ┌──────┬──────┬──────┬──────┬──────┬──────┐
    │ a    ┆ b    ┆ out1 ┆ out2 ┆ out3 ┆ out4 │
    │ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ i64  ┆ i64  ┆ i64  ┆ i64  ┆ f64  │
    ╞══════╪══════╪══════╪══════╪══════╪══════╡
    │ null ┆ null ┆ null ┆ null ┆ null ┆ null │
    │ -1   ┆ -1   ┆ -1   ┆ -1   ┆ -1   ┆ -1.0 │
    │ 0    ┆ 0    ┆ 0    ┆ 0    ┆ 0    ┆ 0.0  │
    │ 1    ┆ 1    ┆ 1    ┆ 1    ┆ 1    ┆ 1.0  │
    │ 2    ┆ 2    ┆ 1    ┆ 2    ┆ 4    ┆ 4.0  │
    └──────┴──────┴──────┴──────┴──────┴──────┘
    ```

    References
    ----------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#signed_powerx-y

    """
    if isinstance(y, (int, float)):
        if y == 1:
            return x.abs() * x.sign()
        elif y == 0:
            return x.sign()
        else:
            return x.abs().pow(y) * x.sign()

    return x.abs().pow(y.cast(Float64)) * x.sign()


def sin(x: Expr) -> Expr:
    """正弦"""
    return x.sin()


def sinh(x: Expr) -> Expr:
    """双曲正弦"""
    return x.sinh()


def softsign(x: Expr) -> Expr:
    """softsign激活函数

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=softsign(pl.col('a')),
    )

    shape: (5, 2)
    ┌──────┬──────────┐
    │ a    ┆ out1     │
    │ ---  ┆ ---      │
    │ i64  ┆ f64      │
    ╞══════╪══════════╡
    │ null ┆ null     │
    │ -1   ┆ -0.5     │
    │ 0    ┆ 0.0      │
    │ 1    ┆ 0.5      │
    │ 2    ┆ 0.666667 │
    └──────┴──────────┘
    ```

    """
    return x / (1 + x.abs())


def sqrt(x: Expr) -> Expr:
    """平方根"""
    return x.sqrt()


def square(x: Expr) -> Expr:
    """平方"""
    return x.pow(2)


def subtract(x: Expr, y: Expr) -> Expr:
    """减法

    x-y"""
    return x - y


def tan(x: Expr) -> Expr:
    """正切

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=tan(pl.col('a')),
    )

    shape: (5, 2)
    ┌──────┬───────────┐
    │ a    ┆ out1      │
    │ ---  ┆ ---       │
    │ i64  ┆ f64       │
    ╞══════╪═══════════╡
    │ null ┆ null      │
    │ -1   ┆ -1.557408 │
    │ 0    ┆ 0.0       │
    │ 1    ┆ 1.557408  │
    │ 2    ┆ -2.18504  │
    └──────┴───────────┘
    ```

    """
    return x.tan()


def tanh(x: Expr) -> Expr:
    """双曲正切

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 0, 1, 2],
    }).with_columns(
        out1=tanh(pl.col('a')),
    )

    shape: (5, 2)
    ┌──────┬───────────┐
    │ a    ┆ out1      │
    │ ---  ┆ ---       │
    │ i64  ┆ f64       │
    ╞══════╪═══════════╡
    │ null ┆ null      │
    │ -1   ┆ -0.761594 │
    │ 0    ┆ 0.0       │
    │ 1    ┆ 0.761594  │
    │ 2    ┆ 0.964028  │
    └──────┴───────────┘
    ```

    """
    return x.tanh()


def var(a: Expr, b: Expr, *args) -> Expr:
    """水平多列求方差

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 1, 1, 2],
        'b': [None, -1, 0, 1, 2],
        'c': [None, -1, 0, 2, None],
    }).with_columns(
        out1=var(pl.col('a'), pl.col('b'), pl.col('c')),
    )

    shape: (5, 4)
    ┌──────┬──────┬──────┬──────────┐
    │ a    ┆ b    ┆ c    ┆ out1     │
    │ ---  ┆ ---  ┆ ---  ┆ ---      │
    │ i64  ┆ i64  ┆ i64  ┆ f64      │
    ╞══════╪══════╪══════╪══════════╡
    │ null ┆ null ┆ null ┆ 0.0      │
    │ -1   ┆ -1   ┆ -1   ┆ 0.0      │
    │ 1    ┆ 0    ┆ 0    ┆ 0.666667 │
    │ 1    ┆ 1    ┆ 2    ┆ 0.666667 │
    │ 2    ┆ 2    ┆ null ┆ 0.0      │
    └──────┴──────┴──────┴──────────┘
    ```

    """
    _args = [a, b] + list(args)
    _mean = mean_horizontal(_args)
    return sum_horizontal([(expr - _mean) ** 2 for expr in _args])


def std(a: Expr, b: Expr, *args) -> Expr:
    """水平多列求标准差

    Examples
    --------
    ```python
    df = pl.DataFrame({
        'a': [None, -1, 1, 1, 2],
        'b': [None, -1, 0, 1, 2],
        'c': [None, -1, 0, 2, None],
    }).with_columns(
        out2=std(pl.col('a'), pl.col('b'), pl.col('c')),
    )

    shape: (5, 4)
    ┌──────┬──────┬──────┬──────────┐
    │ a    ┆ b    ┆ c    ┆ out2     │
    │ ---  ┆ ---  ┆ ---  ┆ ---      │
    │ i64  ┆ i64  ┆ i64  ┆ f64      │
    ╞══════╪══════╪══════╪══════════╡
    │ null ┆ null ┆ null ┆ 0.0      │
    │ -1   ┆ -1   ┆ -1   ┆ 0.0      │
    │ 1    ┆ 0    ┆ 0    ┆ 0.816497 │
    │ 1    ┆ 1    ┆ 2    ┆ 0.816497 │
    │ 2    ┆ 2    ┆ null ┆ 0.0      │
    └──────┴──────┴──────┴──────────┘
    ```

    """
    return var(a, b, *args).sqrt()
