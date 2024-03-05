from typing import List

import numpy as np
from polars import Expr, Series, map_batches

from polars_ta import TA_EPSILON


def cs_standardize_zscore(x: Expr, ddof: int = 0) -> Expr:
    return (x - x.mean()) / x.std(ddof=ddof)


def cs_standardize_minmax(x: Expr) -> Expr:
    a = x.min()
    b = x.max()
    return (x - a) / (b - a + TA_EPSILON)


def cs_winsorize_quantile(x: Expr, low_limit: float = 0.025, up_limit: float = 0.995) -> Expr:
    a = x.quantile(low_limit)
    b = x.quantile(up_limit)
    return x.clip(lower_bound=a, upper_bound=b)


def cs_winsorize_3sigma(x: Expr, n: float = 3.) -> Expr:
    # fill_nan(None) 严重拖慢速度，所以还是由用户自己处理更合适
    a = x.mean()
    b = n * x.std(ddof=0)
    return x.clip(lower_bound=a - b, upper_bound=a + b)


def cs_winsorize_mad(x: Expr, n: float = 3., k: float = 1.4826) -> Expr:
    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    a = x.median()
    b = (n * k) * (x - a).abs().median()
    return x.clip(lower_bound=a - b, upper_bound=a + b)


def cs_neutralize_demean(x: Expr) -> Expr:
    return x - x.mean()


def cs_neutralize_residual_simple(y: Expr, x: Expr) -> Expr:
    """一元回归"""
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


def residual_multiple(cols: List[Series], add_constant: bool) -> Series:
    # https://stackoverflow.com/a/74906705/1894479
    # 比struct.unnest要快一些
    cols = [c.to_numpy() for c in cols]
    if add_constant:
        cols += [np.ones_like(cols[0])]
    yx = np.vstack(cols).T

    # 跳过nan
    mask = np.any(np.isnan(yx), axis=1)
    yx_ = yx[~mask, :]

    y = yx_[:, 0]
    x = yx_[:, 1:]
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    y_hat = np.sum(x * coef, axis=1)
    residual = y - y_hat

    # 回填时，处理空值
    out = np.empty_like(yx[:, 0])
    out[~mask] = residual
    out[mask] = np.nan
    return Series(out, nan_to_null=True)


def cs_neutralize_residual_multiple(y: Expr, *more_x: Expr) -> Expr:
    """多元回归

    Examples
    --------
    >>> cs_neutralize_residual_multiple(EP, LOG_MKT_CAP, *cs.expand_selector(df, cs.matches(r"^sw_l1_\d+$")), ONE)

    Notes
    -----
    1. 常量1，可以通过多输入1列来完成
    2. 正则表达式的用法比较特别，需用`cs.expand_selector`
    """
    return map_batches([y, *more_x], lambda xx: residual_multiple(xx, False))
