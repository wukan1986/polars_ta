import polars as pl


def abs_(x: pl.Expr) -> pl.Expr:
    return x.abs()


def add(*args, filter_=False):
    """Add all inputs (at least 2 inputs required). If filter = true, filter all input NaN to 0 before adding"""
    if filter_:
        # TODO 等官方修复此bug
        # return pl.sum_horizontal(*args)
        exprs = [pl.lit(0)] + list(args)
        return pl.reduce(function=lambda acc, x: acc + x.fill_null(0), exprs=exprs)
    else:
        return pl.reduce(function=lambda acc, x: acc + x, exprs=args)


def ceiling(x: pl.Expr) -> pl.Expr:
    return x.ceil()


def densify(x: pl.Expr) -> pl.Expr:
    raise


def divide(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return x / y


def exp(x: pl.Expr) -> pl.Expr:
    return x.exp()


def floor(x: pl.Expr) -> pl.Expr:
    return x.floor()


def fraction(x: pl.Expr) -> pl.Expr:
    """This operator removes the whole number part and returns the remaining fraction part with sign."""
    # return sign(x) * (abs(x) - floor(abs(x)))
    # return x.sign() * (x.abs() % 1)
    return x % 1


def inverse(x: pl.Expr) -> pl.Expr:
    """1 / x"""
    return 1 / x


def log(x: pl.Expr) -> pl.Expr:
    return x.log()


def log10(x: pl.Expr) -> pl.Expr:
    return x.log10()


def log1p(x: pl.Expr) -> pl.Expr:
    return x.log1p()


def log_diff(x: pl.Expr, d: int = 1) -> pl.Expr:
    """Returns log(current value of input or x[t] ) - log(previous value of input or x[t-1])."""
    return x.log().diff(d)


def max_(*args):
    """Maximum value of all inputs. At least 2 inputs are required."""
    return pl.max_horizontal(args)


def min_(*args):
    """Maximum value of all inputs. At least 2 inputs are required."""
    return pl.min_horizontal(args)


def multiply(*args, filter_=False):
    """Multiply all inputs. At least 2 inputs are required. Filter sets the NaN values to 1"""
    if filter_:
        exprs = [pl.lit(1)] + list(args)
        return pl.reduce(function=lambda acc, x: acc * x.fill_null(1), exprs=exprs)
    else:
        return pl.reduce(function=lambda acc, x: acc * x, exprs=args)


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
    return x.round()


def round_down(x: pl.Expr, f: int = 1) -> pl.Expr:
    """Round input to greatest multiple of f less than input;"""
    if f == 1:
        return x // 1
    else:
        return x // f * f


def s_log_1p(x: pl.Expr) -> pl.Expr:
    return (1 + x.abs()).log() * x.sign()


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


def sqrt(x: pl.Expr) -> pl.Expr:
    return x.sqrt()


def subtract(*args, filter_=False):
    """x-y. If filter = true, filter all input NaN to 0 before subtracting"""
    if filter_:
        exprs = [pl.lit(0)] + list(args)
        return pl.reduce(function=lambda acc, x: acc - x.fill_null(0), exprs=exprs)
    else:
        return pl.reduce(function=lambda acc, x: acc - x, exprs=args)
