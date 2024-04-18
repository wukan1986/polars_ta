import polars_ols as pls
from polars import Expr, when
from polars_ols import OLSKwargs

# In the original version, the function names are not prefixed with `cs_`,
# here we add it to prevent confusion
# 原版函数名都没有加`cs_`, 这里统一加一防止混淆


_ols_kwargs = OLSKwargs(null_policy='drop', solve_method='svd')


def cs_one_side(x: Expr, is_long: bool = True) -> Expr:
    """Shifts all instruments up or down so that the Alpha becomes long-only or short-only
(if side = short), respectively."""
    # TODO: 这里不确定，需再研究
    # [-1, 0, 1]+1=[0, 1, 2]
    # max([-1, 0, 1], 0)=[0,0,1]
    raise


def cs_rank(x: Expr, pct: bool = True) -> Expr:
    """Ranks the input among all the instruments and returns an equally distributed number between 0.0 and 1.0. For precise sort, use the rate as 0."""
    if pct:
        return x.rank(method='min') / x.count()
    else:
        return x.rank(method='min')


def cs_scale(x: Expr, scale_: float = 1, long_scale: float = 1, short_scale: float = 1) -> Expr:
    """Scales input to booksize. We can also scale the long positions and short positions to separate scales by mentioning additional parameters to the operator."""
    if long_scale != 1 or short_scale != 1:
        L = x.clip(lower_bound=0)  # 全正数
        S = x.clip(upper_bound=0)  # 全负数，和还是负数
        return L / L.sum() * long_scale - S / S.sum() * short_scale
    else:
        return x / x.abs().sum() * scale_


def cs_truncate(x: Expr, max_percent: float = 0.01) -> Expr:
    """Operator truncates all values of x to maxPercent. Here, maxPercent is in decimal notation."""
    return x.clip(upper_bound=x.sum() * max_percent)


def cs_fill_zero(x: Expr) -> Expr:
    """截面不全为空时，空值填充为0，反之保持null

    在权重矩阵中使用时。一定要保证所有股票都在，停牌不能被过滤了"""
    return when(x.is_not_null().sum() == 0).then(x).otherwise(x.fill_null(0))


def cs_regression_neut(y: Expr, x: Expr) -> Expr:
    """一元回归残差"""
    return pls.compute_least_squares(y, x, add_intercept=True, mode='residuals', ols_kwargs=_ols_kwargs)


def cs_regression_proj(y: Expr, x: Expr) -> Expr:
    """一元回归预测"""
    return pls.compute_least_squares(y, x, add_intercept=True, mode='predictions', ols_kwargs=_ols_kwargs)
