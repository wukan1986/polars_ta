import numpy as np
from polars import Expr, Series, fold, any_horizontal
from polars import max_horizontal, sum_horizontal, min_horizontal, mean_horizontal


def abs_(x: Expr) -> Expr:
    """绝对值"""
    if isinstance(x, (Expr, Series)):
        return x.abs()
    else:
        return np.abs(x)


def add(a: Expr, b: Expr, *args) -> Expr:
    """水平多列加

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


def ceiling(x: Expr) -> Expr:
    """向上取整"""
    return x.ceil()


def cos(x: Expr) -> Expr:
    """余弦"""
    return x.cos()


def cosh(x: Expr) -> Expr:
    """双曲余弦"""
    return x.cosh()


def densify(x: Expr) -> Expr:
    raise


def divide(x: Expr, y: Expr) -> Expr:
    """x/y"""
    return x / y


def exp(x: Expr) -> Expr:
    """自然指数函数"""
    return x.exp()


def expm1(x: Expr) -> Expr:
    """对数收益率 转 简单收益率 convert log return to simple return"""
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
    """1/x"""
    return 1 / x


def log(x: Expr) -> Expr:
    """e为底的对数"""
    if isinstance(x, (Expr, Series)):
        return x.log()
    else:
        return np.log(x)


def log10(x: Expr) -> Expr:
    """10为底的对数"""
    return x.log10()


def log1p(x: Expr) -> Expr:
    """简单收益率 转 对数收益率 convert simple return to log return

    log(x+1)
    """
    return x.log1p()


def max_(a: Expr, b: Expr, *args) -> Expr:
    """水平多列最大值 Maximum value of all inputs. At least 2 inputs are required."""
    return max_horizontal(a, b, *args)


def mean(a: Expr, b: Expr, *args) -> Expr:
    """水平多列均值"""
    return mean_horizontal(a, b, *args)


def min_(a: Expr, b: Expr, *args) -> Expr:
    """水平多列最小值 Maximum value of all inputs. At least 2 inputs are required."""
    return min_horizontal(a, b, *args)


def mod(x: Expr, y: Expr) -> Expr:
    """x%y"""
    return x % y


def multiply(a: Expr, b: Expr, *args) -> Expr:
    """水平多列乘 Multiply all inputs. At least 2 inputs are required.

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
    """x ** y"""
    return x.pow(y)


def reverse(x: Expr) -> Expr:
    """-x"""
    return -x


def round_(x: Expr, decimals: int = 0) -> Expr:
    """四舍五入 Round input to closest integer.

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
    """小于输入的f的最大倍数 Round input to greatest multiple of f less than input

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
    """符号"""
    if isinstance(x, (Expr, Series)):
        return x.sign()
    else:
        return np.sign(x)


def signed_power(x: Expr, y: Expr) -> Expr:
    """x raised to the power of y such that final result preserves sign of x.


    References
    ----------
    https://platform.worldquantbrain.com/learn/operators/detailed-operator-descriptions#signed_powerx-y

    """
    if isinstance(y, (int, float)):
        if y == 1:
            return x.abs() * x.sign()
        elif y == 0:
            return x.sign()

    return x.abs().pow(y) * x.sign()


def sin(x: Expr) -> Expr:
    """正弦"""
    return x.sin()


def sinh(x: Expr) -> Expr:
    """双曲正弦"""
    return x.sinh()


def softsign(x: Expr) -> Expr:
    """softsign是 tanh激活函数的另一个替代选择"""
    return x / (1 + x.abs())


def sqrt(x: Expr) -> Expr:
    """平方根"""
    return x.sqrt()


def subtract(x: Expr, y: Expr) -> Expr:
    """x-y"""
    return x - y


def tan(x: Expr) -> Expr:
    """正切"""
    return x.tan()


def tanh(x: Expr) -> Expr:
    """双曲正切"""
    return x.tanh()


def var(a: Expr, b: Expr, *args) -> Expr:
    """水平多列方差"""
    _args = [a, b] + list(args)
    _mean = mean_horizontal(_args)
    _sum = sum_horizontal([(expr - _mean) ** 2 for expr in _args])
    return _sum


def std(a: Expr, b: Expr, *args) -> Expr:
    """水平多列标准差"""
    return var(a, b, *args).sqrt()
