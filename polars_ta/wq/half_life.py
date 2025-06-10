import numpy as np
from polars import Expr

from polars_ta.utils.numba_ import get_exponent_weights


def ts_mean_hl(x: Expr, d: int, half_life: int):
    """滚动均值。带半衰期"""
    return x.fill_null(np.nan).rolling_mean(d, weights=get_exponent_weights(d, half_life)).fill_nan(None)


def ts_sum_hl(x: Expr, d: int, half_life: int):
    """滚动求和。带半衰期"""
    return x.fill_null(np.nan).rolling_sum(d, weights=get_exponent_weights(d, half_life)).fill_nan(None)


def ts_std_hl(x: Expr, d: int, half_life: int):
    """滚动标准差。带半衰期"""
    return x.fill_null(np.nan).rolling_std(d, weights=get_exponent_weights(d, half_life)).fill_nan(None)


def ts_var_hl(x: Expr, d: int, half_life: int):
    """滚动方差。带半衰期"""
    return x.fill_null(np.nan).rolling_var(d, weights=get_exponent_weights(d, half_life)).fill_nan(None)

# TODO 混动时序回归，带半衰期
