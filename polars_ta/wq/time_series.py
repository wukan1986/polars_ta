import numpy as np
import polars as pl


def ts_arg_max(x: pl.Expr, d: int = 5) -> pl.Expr:
    # WorldQuant中最大值为今天返回0，为昨天返回1
    return d - 1 - x.rolling_map(np.nanargmax, d)


def ts_arg_min(x: pl.Expr, d: int = 5) -> pl.Expr:
    return d - 1 - x.rolling_map(np.nanargmax, d)


def ts_delay(x: pl.Expr, d: int = 1) -> pl.Expr:
    return x.shift(d)


def ts_delta(x: pl.Expr, d: int = 1) -> pl.Expr:
    return x.diff(d)


def ts_max(x: pl.Expr, d: int = 30) -> pl.Expr:
    return x.rolling_max(d)


def ts_mean(x: pl.Expr, d: int = 5) -> pl.Expr:
    return x.rolling_mean(d)


def ts_median(x: pl.Expr, d: int = 5) -> pl.Expr:
    return x.rolling_median(d)


def ts_min(x: pl.Expr, d: int = 30) -> pl.Expr:
    return x.rolling_min(d)


def ts_min_diff(x: pl.Expr, d: int = 30) -> pl.Expr:
    return x - ts_min(x, d)


def ts_returns(x: pl.Expr, d: int = 1) -> pl.Expr:
    return x.pct_change(d)


def ts_std_dev(x: pl.Expr, d: int = 5, ddof: int = 0) -> pl.Expr:
    return x.rolling_std(d, ddof=ddof)


def ts_scale(x: pl.Expr, d: int = 5) -> pl.Expr:
    a = ts_min(x, d)
    b = ts_max(x, d)
    return (x - a) / (b - a)


def ts_sum(x: pl.Expr, d: int = 30) -> pl.Expr:
    return x.rolling_sum(d)


def ts_zscore(x: pl.Expr, d: int = 5) -> pl.Expr:
    return (x - ts_mean(x, d)) / ts_std_dev(x, d)
