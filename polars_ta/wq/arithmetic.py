import polars as pl


def abs_(x: pl.Expr) -> pl.Expr:
    return x.abs()


def ceiling(x: pl.Expr) -> pl.Expr:
    return x.ceil()


def exp(x: pl.Expr) -> pl.Expr:
    return x.exp()


def floor(x: pl.Expr) -> pl.Expr:
    return x.floor()


def fraction(x: pl.Expr) -> pl.Expr:
    """This operator removes the whole number part and returns the remaining fraction part with sign."""
    # return sign(x) * (abs(x) - floor(abs(x)))
    return x.sign() * (x.abs() % 1)


def inverse(x: pl.Expr) -> pl.Expr:
    """1 / x"""
    return 1 / x


def log(x: pl.Expr) -> pl.Expr:
    return x.log()


def log_diff(x: pl.Expr, d: int = 1) -> pl.Expr:
    """Returns log(current value of input or x[t] ) - log(previous value of input or x[t-1])."""
    return x.log().diff(d)


def power(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """x ^ y"""
    return x.pow(y)


def purify(x: pl.Expr) -> pl.Expr:
    """Clear infinities (+inf, -inf) by replacing with NaN."""
    return pl.when(x.is_infinite()).then(pl.Null).otherwise(x)


def reverse(x: pl.Expr) -> pl.Expr:
    """- x"""
    return -x


def round_(x: pl.Expr) -> pl.Expr:
    """Round input to closest integer."""
    return x.round(decimals=0)


def round_down(x: pl.Expr, f: int = 1) -> pl.Expr:
    """Round input to greatest multiple of f less than input;"""
    if f == 1:
        return x // f
    else:
        return x // f * f


def sign(x: pl.Expr) -> pl.Expr:
    return x.sign()


def signed_power(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """x raised to the power of y such that final result preserves sign of x."""
    if isinstance(y, (int, float)):
        if y == 1:
            return x.abs() * x.sign()
        elif y == 0:
            return x.sign()

    return x.abs().pow(y) * x.sign()


def scale(x: pl.Expr, scale_: float = 1) -> pl.Expr:
    if scale_ == 1:
        # TODO 返回的是表达式，还未开始计算，少一步是否可以加速？
        return x / x.abs().sum()
    else:
        return x / x.abs().sum() * scale_


def log(x: pl.Expr) -> pl.Expr:
    return x.log()


def log10(x: pl.Expr) -> pl.Expr:
    return x.log10()


def log1p(x: pl.Expr) -> pl.Expr:
    return x.log1p()


def s_log_1p(x: pl.Expr) -> pl.Expr:
    return (1 + x.abs()).log() * x.sign()


def sqrt(x: pl.Expr) -> pl.Expr:
    return x.sqrt()
