import polars_ols as pls
from polars import Expr, when
from polars_ols.least_squares import OLSKwargs

from polars_ta.wq.cross_sectional import cs_rank


# ======================
# standardize
def cs_zscore(x: Expr, ddof: int = 0) -> Expr:
    return (x - x.mean()) / x.std(ddof=ddof)


def cs_minmax(x: Expr) -> Expr:
    a = x.min()
    b = x.max()
    # 这个版本在b-a为整数时，得到的结果不好看
    # return (x - a) / (b - a + TA_EPSILON)
    return when(a != b).then((x - a) / (b - a)).otherwise(0)


# ======================
# winsorize
def cs_quantile(x: Expr, low_limit: float = 0.025, up_limit: float = 0.995) -> Expr:
    a = x.quantile(low_limit)
    b = x.quantile(up_limit)
    return x.clip(lower_bound=a, upper_bound=b)


def cs_3sigma(x: Expr, n: float = 3.) -> Expr:
    # fill_nan will seriously reduce speed. So it's more appropriate for users to handle it themselves
    # fill_nan(None) 严重拖慢速度，所以还是由用户自己处理更合适
    a = x.mean()
    b = n * x.std(ddof=0)
    return x.clip(lower_bound=a - b, upper_bound=a + b)


def cs_mad(x: Expr, n: float = 3., k: float = 1.4826) -> Expr:
    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    a = x.median()
    b = (n * k) * (x - a).abs().median()
    return x.clip(lower_bound=a - b, upper_bound=a + b)


# ======================
# neutralize
def cs_demean(x: Expr) -> Expr:
    """demean

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


# def _residual_multiple(cols: List[Series], add_constant: bool) -> Series:
#     # 将pl.Struct转成list,这样可以实现传正则，其它也转list
#     cols = [list(c.struct) if isinstance(c.dtype, Struct) else [c] for c in cols]
#     # 二维列表转一维列表，再转np.ndarray
#     cols = [i.to_numpy() for p in cols for i in p]
#     if add_constant:
#         cols += [np.ones_like(cols[0])]
#     yx = np.vstack(cols).T
#
#     # skip nan
#     mask = np.any(np.isnan(yx), axis=1)
#     yx_ = yx[~mask, :]
#
#     y = yx_[:, 0]
#     x = yx_[:, 1:]
#     coef = np.linalg.lstsq(x, y, rcond=None)[0]
#     y_hat = np.sum(x * coef, axis=1)
#     residual = y - y_hat
#
#     # refill
#     out = np.empty_like(yx[:, 0])
#     out[~mask] = residual
#     out[mask] = np.nan
#     return Series(out, nan_to_null=True)
#
#
# def cs_resid_(y: Expr, *more_x: Expr) -> Expr:
#     """multivariate regression
#     多元回归
#     """
#     return map_batches([y, *more_x], lambda xx: _residual_multiple(xx, False))


def cs_resid(y: Expr, *more_x: Expr) -> Expr:
    """多元回归取残差"""
    return pls.compute_least_squares(y, *more_x, mode='residuals', ols_kwargs=_ols_kwargs)


def cs_mad_zscore(y: Expr) -> Expr:
    """去极值、标准化"""
    return cs_zscore(cs_mad(y))


def cs_mad_zscore_resid(y: Expr, *more_x: Expr) -> Expr:
    """去极值、标准化、中性化"""
    return cs_resid(cs_zscore(cs_mad(y)), *more_x)


def cs_mad_rank(y: Expr) -> Expr:
    """去极值，排名。"""
    return cs_rank(cs_mad(y))


def cs_mad_rank2(y: Expr, m: float) -> Expr:
    """非线性处理。去极值，排名，移动峰或谷到零点，然后平方

    适合于分层收益V型或倒V的情况"""
    return (cs_rank(cs_mad(y)) - m) ** 2


def cs_mad_rank2_resid(y: Expr, m: float, *more_x: Expr) -> Expr:
    """非线性处理。去极值，排名，移动峰或谷到零点，然后平方。回归取残差

    适合于分层收益V型或倒V的情况"""
    return cs_resid((cs_rank(cs_mad(y)) - m) ** 2, *more_x)


def cs_rank2(y: Expr, m: float) -> Expr:
    """非线性处理。移动峰或谷到零点，然后平方

    适合于分层收益V型或倒V的情况"""
    return (cs_rank(y) - m) ** 2
