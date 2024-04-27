from polars import Expr

from polars_ta.wq import ts_corr, ts_zscore, ts_std_dev, ts_regression_slope


def ts_RSRS_R2(high: Expr, low: Expr, n: int = 18, m: int = 600) -> Expr:
    """光大RSRS指标，中金QRS指标，R^2调整

    References
    ----------
    中金：金融工程视角下的技术择时艺术
    """
    a = ts_corr(high, low, n)
    return ts_zscore(ts_std_dev(high, n) / ts_std_dev(low, n) * a, m) * (a ** 2)


def ts_RSRS(high: Expr, low: Expr, n: int = 18, m: int = 600) -> Expr:
    """光大RSRS指标，中金QRS指标

    References
    ----------
    中金：金融工程视角下的技术择时艺术
    """
    return ts_zscore(ts_regression_slope(high, low, n), m)
