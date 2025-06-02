"""
补空值 → 去极值 → 标准化 → 中性化 → 标准化（可选二次标准化）

# 对数市值。去极值
MC_LOG = cs_quantile(log1p(market_cap), 0.01, 0.99)
# 对数市值。标准化。供其他因子市值中性化时使用
MC_NORM = cs_zscore(MC_LOG)
# 对数市值。行业中性化。直接作为因子使用
MC_NEUT = cs_zscore(cs_resid(MC_NORM, CS_SW_L1, ONE))

"""
import polars_ols as pls
from polars import Expr, when
from polars_ols.least_squares import OLSKwargs


# ======================
# standardize
def cs_zscore(x: Expr, ddof: int = 0) -> Expr:
    """横截面zscore标准化"""
    return (x - x.mean()) / x.std(ddof=ddof)


def cs_minmax(x: Expr) -> Expr:
    """横截面minmax标准化"""
    a = x.min()
    b = x.max()
    # 这个版本在b-a为整数时，得到的结果不好看
    # return (x - a) / (b - a + TA_EPSILON)
    return when(a != b).then((x - a) / (b - a)).otherwise(0)


# ======================
# winsorize
def cs_quantile(x: Expr, low_limit: float = 0.025, up_limit: float = 0.975) -> Expr:
    """横截面分位数去极值"""
    a = x.quantile(low_limit)
    b = x.quantile(up_limit)
    return x.clip(lower_bound=a, upper_bound=b)


def cs_3sigma(x: Expr, n: float = 3.) -> Expr:
    """横截面3倍sigma去极值"""
    # fill_nan will seriously reduce speed. So it's more appropriate for users to handle it themselves
    # fill_nan(None) 严重拖慢速度，所以还是由用户自己处理更合适
    a = x.mean()
    b = n * x.std(ddof=0)
    return x.clip(lower_bound=a - b, upper_bound=a + b)


def cs_mad(x: Expr, n: float = 3., k: float = 1.4826) -> Expr:
    """横截面MAD去极值

    References
    ----------
    https://en.wikipedia.org/wiki/Median_absolute_deviation

    """
    a = x.median()
    b = (n * k) * (x - a).abs().median()
    return x.clip(lower_bound=a - b, upper_bound=a + b)


# ======================
# neutralize
def cs_demean(x: Expr) -> Expr:
    """横截面去均值化

    Notes
    -----
    Slower than multivariate regression. We need to groupby date and industry here,
    while multivariate regression only needs to add industry dummy variables and then groupby date

    Notes
    -----
    速度没有多元回归快，因为这里需要按日期行业groupby，
    而多元回归只要添加行业哑变量，然后按日期groupby即可

    """
    return x - x.mean()


# ======================
# neutralize
_ols_kwargs = OLSKwargs(null_policy='drop', solve_method='svd')


def cs_resid(y: Expr, *more_x: Expr) -> Expr:
    """横截面多元回归取残差"""
    return pls.compute_least_squares(y, *more_x, mode='residuals', ols_kwargs=_ols_kwargs)


def cs_zscore_resid(y: Expr, *more_x: Expr) -> Expr:
    """横截面标准化、中性化"""
    return cs_resid(cs_zscore(y), *more_x)


def cs_mad_zscore(y: Expr) -> Expr:
    """横截面去极值、标准化"""
    return cs_zscore(cs_mad(y))


def cs_mad_zscore_resid(y: Expr, *more_x: Expr) -> Expr:
    """横截面去极值、标准化、中性化"""
    return cs_resid(cs_zscore(cs_mad(y)), *more_x)


def cs_mad_zscore_resid_zscore(y: Expr, *more_x: Expr) -> Expr:
    """横截面去极值、标准化、中性化、二次标准化"""
    return cs_zscore(cs_resid(cs_zscore(cs_mad(y)), *more_x))
