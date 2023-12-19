from typing import Sequence

import numpy as np
import polars as pl


def standardize_zscore(x: pl.Expr, ddof: int = 0) -> pl.Expr:
    return (x - x.mean()) / x.std(ddof=ddof)


def standardize_minmax(x: pl.Expr) -> pl.Expr:
    a = x.min()
    b = x.max()
    return (x - a) / (b - a)


def winsorize_quantile(x: pl.Expr, low_limit: float = 0.025, up_limit: float = 0.995) -> pl.Expr:
    a = x.quantile(low_limit)
    b = x.quantile(up_limit)
    return x.clip(lower_bound=a, upper_bound=b)


def winsorize_3sigma(x: pl.Expr, n: float = 3.) -> pl.Expr:
    a = x.mean()
    b = n * x.std(ddof=0)
    return x.clip(lower_bound=a - b, upper_bound=a + b)


def winsorize_mad(x: pl.Expr, n: float = 3., k: float = 1.4826) -> pl.Expr:
    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    a = x.median()
    b = (n * k) * (x - a).abs().median()
    return x.clip(lower_bound=a - b, upper_bound=a + b)


def neutralize_demean(x: pl.Expr) -> pl.Expr:
    return x - x.mean()


def neutralize_residual_simple(y: pl.Expr, x: pl.Expr) -> pl.Expr:
    # https://stackoverflow.com/a/74906705/1894479
    # 一元回归时，这个版本更快，不需再补充常量1
    # e_i = y_i - a - bx_i
    #     = y_i - ȳ + bx̄ - bx_i
    #     = y_i - ȳ - b(x_i - x̄)
    x_demeaned = x - x.mean()
    y_demeaned = y - y.mean()
    x_demeaned_squared = x_demeaned.pow(2)
    beta = x_demeaned.dot(y_demeaned) / x_demeaned_squared.sum()
    return y_demeaned - beta * x_demeaned


def neutralize_residual_multiple(cols: Sequence[pl.Series]) -> pl.Series:
    # https://stackoverflow.com/a/74906705/1894479
    # 比struct.unnest要快一些
    cols = [c.to_numpy() for c in cols]
    y = cols[0]
    x = cols[1:]
    A = np.vstack(x).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = np.sum(A * coef, axis=1)
    residual = y - y_hat
    return pl.Series(residual)
