from polars import Expr

from polars_ta.wq.preprocess import neutralize_demean
from polars_ta.wq.preprocess import standardize_minmax
from polars_ta.wq.preprocess import standardize_zscore
from polars_ta.wq.preprocess import winsorize_3sigma


# 原版函数名都没有加`cs_`, 这里统一加一防止混淆

def cs_normalize(x: Expr, use_std: bool = False, limit: float = 0.0) -> Expr:
    """Calculates the mean value of all valid alpha values for a certain date, then subtracts that mean from each element."""
    if use_std:
        # 这里用ddof=1才能与文档示例的数值对应上
        r = standardize_zscore(x, ddof=1)
    else:
        r = neutralize_demean(x)

    if limit == 0:
        return r
    else:
        return r.clip(-limit, limit)


def cs_one_side(x: Expr, is_long: bool = True) -> Expr:
    """Shifts all instruments up or down so that the Alpha becomes long-only or short-only
(if side = short), respectively."""
    # TODO: 这里不确定，需再研究
    # [-1, 0, 1]+1=[0, 1, 2]
    # max([-1, 0, 1], 0)=[0,0,1]
    raise


def cs_rank(x: Expr, rate: int = 2, pct: bool = True) -> Expr:
    """Ranks the input among all the instruments and returns an equally distributed number between 0.0 and 1.0. For precise sort, use the rate as 0."""
    if pct:
        return x.rank() / x.count()
    else:
        return x.rank()


def cs_scale(x: Expr, scale_: float = 1, long_scale: float = 1, short_scale: float = 1) -> Expr:
    """Scales input to booksize. We can also scale the long positions and short positions to separate scales by mentioning additional parameters to the operator."""
    if long_scale != 1 or short_scale != 1:
        L = x.clip(lower_bound=0)  # 全正数
        S = x.clip(upper_bound=0)  # 全负数，和还是负数
        return L / L.sum() * long_scale - S / S.sum() * short_scale
    else:
        return x / x.abs().sum() * scale_


def cs_scale_down(x: Expr) -> Expr:
    """Scales all values in each day proportionately between 0 and 1 such that minimum value maps to 0 and maximum value maps to 1. Constant is the offset by which final result is subtracted."""
    return standardize_minmax(x)


def cs_truncate(x: Expr, max_percent: float = 0.01) -> Expr:
    """Operator truncates all values of x to maxPercent. Here, maxPercent is in decimal notation."""
    return x.clip(upper_bound=x.sum() * max_percent)


def cs_winsorize(x: Expr, std: float = 4) -> Expr:
    return winsorize_3sigma(x, std)


def cs_zscore(x: Expr) -> Expr:
    return standardize_zscore(x)
