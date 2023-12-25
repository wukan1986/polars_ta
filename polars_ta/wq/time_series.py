import numpy as np
import polars as pl

from polars_ta.utils.pandas_ import roll_kurt
from polars_ta.utils.pandas_ import roll_rank


# TODO rolling_map比较慢，少用. 如ts_arg_max、ts_product等

def ts_arg_max(x: pl.Expr, d: int = 5) -> pl.Expr:
    # WorldQuant中最大值为今天返回0，为昨天返回1
    return d - 1 - x.rolling_map(np.argmax, d)


def ts_arg_min(x: pl.Expr, d: int = 5) -> pl.Expr:
    return d - 1 - x.rolling_map(np.argmax, d)


def ts_co_kurtosis(x: pl.Expr, y: pl.Expr, d: int = 5, ddof: int = 1) -> pl.Expr:
    raise


def ts_co_skewness(x: pl.Expr, y: pl.Expr, d: int = 5, ddof: int = 1) -> pl.Expr:
    raise


def ts_corr(x: pl.Expr, y: pl.Expr, d: int = 5, ddof: int = 1) -> pl.Expr:
    # x、y不区分先后
    return pl.rolling_corr(x, y, window_size=d, ddof=ddof)


def ts_count(x: pl.Expr, d: int = 30) -> pl.Expr:
    return x.cast(pl.Int32).rolling_sum(d)


def ts_count_nans(x: pl.Expr, d: int = 5) -> pl.Expr:
    # null与nan到底用哪一个？
    return x.is_null().rolling_sum(d)


def ts_covariance(x: pl.Expr, y: pl.Expr, d: int = 5, ddof: int = 1) -> pl.Expr:
    # x、y不区分先后
    return pl.rolling_cov(x, y, window_size=d, ddof=ddof)


def ts_decay_exp_window(x: pl.Expr, d: int = 30, factor: float = 1.0) -> pl.Expr:
    # TODO weights not yet supported on array with null values
    y = pl.arange(d - 1, -1, step=-1, eager=False)
    weights = pl.repeat(factor, d, eager=True).pow(y)
    return x.rolling_mean(d, weights=weights)


def ts_decay_linear(x: pl.Expr, d: int = 30, dense: bool = False) -> pl.Expr:
    # TODO weights not yet supported on array with null values
    weights = pl.arange(1, d + 1, eager=True)
    return x.rolling_mean(d, weights=weights)


def ts_delay(x: pl.Expr, d: int = 1) -> pl.Expr:
    return x.shift(d)


def ts_delta(x: pl.Expr, d: int = 1) -> pl.Expr:
    return x.diff(d)


def ts_ir(x: pl.Expr, d: int = 1) -> pl.Expr:
    return ts_mean(x, d) / ts_std_dev(x, d)


def ts_kurtosis(x: pl.Expr, d: int = 5) -> pl.Expr:
    # TODO 等待polars官方出rolling_kurt
    return x.map_batches(lambda a: roll_kurt(a, d))


def ts_max(x: pl.Expr, d: int = 30) -> pl.Expr:
    return x.rolling_max(d)


def ts_max_diff(x: pl.Expr, d: int = 30) -> pl.Expr:
    return x - ts_max(x, d)


def ts_mean(x: pl.Expr, d: int = 5) -> pl.Expr:
    return x.rolling_mean(d)


def ts_median(x: pl.Expr, d: int = 5) -> pl.Expr:
    return x.rolling_median(d)


def ts_min(x: pl.Expr, d: int = 30) -> pl.Expr:
    return x.rolling_min(d)


def ts_min_diff(x: pl.Expr, d: int = 30) -> pl.Expr:
    return x - ts_min(x, d)


def ts_product(x: pl.Expr, d: int = 5) -> pl.Expr:
    return x.rolling_map(np.nanprod, d)


def ts_rank(x: pl.Expr, d: int = 5, constant=0) -> pl.Expr:
    # TODO 等待polars官方出rolling_rank，并支持pct
    # bottleneck长期无人维护，pydata/bottleneck#434 没有合并
    # pandas中已经用跳表实现了此功能，速度也不差
    return x.map_batches(lambda a: roll_rank(a, d, True))


def ts_returns(x: pl.Expr, d: int = 1) -> pl.Expr:
    return x.pct_change(d)


def ts_scale(x: pl.Expr, d: int = 5) -> pl.Expr:
    a = ts_min(x, d)
    b = ts_max(x, d)
    return (x - a) / (b - a)


def ts_skewness(x: pl.Expr, d: int = 5, bias=False) -> pl.Expr:
    # bias=False与pandas结果一样
    return x.rolling_skew(d, bias=bias)


def ts_std_dev(x: pl.Expr, d: int = 5, ddof: int = 0) -> pl.Expr:
    return x.rolling_std(d, ddof=ddof)


def ts_sum(x: pl.Expr, d: int = 30) -> pl.Expr:
    return x.rolling_sum(d)


def ts_weighted_delay(x: pl.Expr, k: float = 0.5) -> pl.Expr:
    return x.rolling_sum(2, weights=[1 - k, k])


def ts_zscore(x: pl.Expr, d: int = 5) -> pl.Expr:
    return (x - ts_mean(x, d)) / ts_std_dev(x, d)
