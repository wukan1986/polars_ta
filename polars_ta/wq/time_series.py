import numpy as np
from polars import Expr, Series, rolling_corr, Int32, rolling_cov, arange, repeat

from polars_ta.utils.pandas_ import roll_kurt
from polars_ta.utils.pandas_ import roll_rank


# TODO rolling_map比较慢，少用. 如ts_arg_max、ts_product等

def _arg_max(x: Series):
    """
    Notes
    -----
    等polars推出rolling_arg_max(reverse=True)这个问题能好转

    """
    # return x[::-1].arg_max()
    # return x.reverse().arg_max() # 正确，但太慢
    return len(x) - 1 - x.arg_max()  # 有多个最大值相同时，靠前的值会被记录下来，导致结果偏大


def ts_arg_max(x: Expr, d: int = 5) -> Expr:
    # WorldQuant中最大值为今天返回0，为昨天返回1
    return x.rolling_map(_arg_max, d)


def _arg_min(x: Series):
    return len(x) - 1 - x.arg_min()


def ts_arg_min(x: Expr, d: int = 5) -> Expr:
    return x.rolling_map(_arg_min, d)


def ts_co_kurtosis(x: Expr, y: Expr, d: int = 5, ddof: int = 1) -> Expr:
    raise


def ts_co_skewness(x: Expr, y: Expr, d: int = 5, ddof: int = 1) -> Expr:
    raise


def ts_corr(x: Expr, y: Expr, d: int = 5, ddof: int = 1) -> Expr:
    # x、y不区分先后
    return rolling_corr(x, y, window_size=d, ddof=ddof)


def ts_count(x: Expr, d: int = 30) -> Expr:
    return x.cast(Int32).rolling_sum(d)


def ts_count_nans(x: Expr, d: int = 5) -> Expr:
    # null与nan到底用哪一个？
    return x.is_null().rolling_sum(d)


def ts_covariance(x: Expr, y: Expr, d: int = 5, ddof: int = 1) -> Expr:
    # x、y不区分先后
    return rolling_cov(x, y, window_size=d, ddof=ddof)


def ts_decay_exp_window(x: Expr, d: int = 30, factor: float = 1.0) -> Expr:
    # TODO weights not yet supported on array with null values
    y = arange(d - 1, -1, step=-1, eager=False)
    weights = repeat(factor, d, eager=True).pow(y)
    return x.rolling_mean(d, weights=weights)


def ts_decay_linear(x: Expr, d: int = 30, dense: bool = False) -> Expr:
    # TODO weights not yet supported on array with null values
    weights = arange(1, d + 1, eager=True)
    return x.rolling_mean(d, weights=weights)


def ts_delay(x: Expr, d: int = 1) -> Expr:
    return x.shift(d)


def ts_delta(x: Expr, d: int = 1) -> Expr:
    return x.diff(d)


def ts_ir(x: Expr, d: int = 1) -> Expr:
    return ts_mean(x, d) / ts_std_dev(x, d)


def ts_kurtosis(x: Expr, d: int = 5) -> Expr:
    # TODO 等待polars官方出rolling_kurt
    return x.map_batches(lambda a: roll_kurt(a, d))


def ts_log_diff(x: Expr, d: int = 1) -> Expr:
    """Returns log(current value of input or x[t] ) - log(previous value of input or x[t-1])."""
    return x.log().diff(d)


def ts_max(x: Expr, d: int = 30) -> Expr:
    return x.rolling_max(d)


def ts_max_diff(x: Expr, d: int = 30) -> Expr:
    return x - ts_max(x, d)


def ts_mean(x: Expr, d: int = 5) -> Expr:
    return x.rolling_mean(d)


def ts_median(x: Expr, d: int = 5) -> Expr:
    return x.rolling_median(d)


def ts_min(x: Expr, d: int = 30) -> Expr:
    return x.rolling_min(d)


def ts_min_diff(x: Expr, d: int = 30) -> Expr:
    return x - ts_min(x, d)


def ts_product(x: Expr, d: int = 5) -> Expr:
    return x.rolling_map(np.nanprod, d)


def ts_rank(x: Expr, d: int = 5, constant=0) -> Expr:
    # TODO 等待polars官方出rolling_rank，并支持pct
    # bottleneck长期无人维护，pydata/bottleneck#434 没有合并
    # pandas中已经用跳表实现了此功能，速度也不差
    return x.map_batches(lambda a: roll_rank(a, d, True))


def ts_returns(x: Expr, d: int = 1) -> Expr:
    return x.pct_change(d)


def ts_scale(x: Expr, d: int = 5) -> Expr:
    a = ts_min(x, d)
    b = ts_max(x, d)
    return (x - a) / (b - a)


def ts_skewness(x: Expr, d: int = 5, bias=False) -> Expr:
    # bias=False与pandas结果一样
    return x.rolling_skew(d, bias=bias)


def ts_std_dev(x: Expr, d: int = 5, ddof: int = 0) -> Expr:
    return x.rolling_std(d, ddof=ddof)


def ts_sum(x: Expr, d: int = 30) -> Expr:
    return x.rolling_sum(d)


def ts_weighted_delay(x: Expr, k: float = 0.5) -> Expr:
    return x.rolling_sum(2, weights=[1 - k, k])


def ts_zscore(x: Expr, d: int = 5) -> Expr:
    return (x - ts_mean(x, d)) / ts_std_dev(x, d)
