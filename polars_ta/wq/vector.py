"""
本文件中算子都是对整列计算得到一个值，使用时可能会广播到整列

可以用在横截面，也可以用在时序。但时序使用会引入未来数据，这时一般用途是统计
"""

from polars import Expr


def vec_avg(x: Expr) -> Expr:
    """Taking mean of the vector field x"""
    return x.mean()


def vec_choose(x: Expr, nth: int) -> Expr:
    """Choosing kth item(indexed at 0) from each vector field x"""
    return x.gather(nth)


def vec_count(x: Expr) -> Expr:
    """Number of elements in vector field x"""
    return x.count()


def vec_ir(x: Expr) -> Expr:
    """Information Ratio (Mean / Standard Deviation) of vector field x"""
    return x.mean() / x.std(ddof=0)


def vec_kurtosis(x: Expr) -> Expr:
    """Kurtosis of vector field x"""
    return x.kurtosis()


def vec_l2_norm(x: Expr) -> Expr:
    """Euclidean norm"""
    return x.pow(2).sum().sqrt()


def vec_max(x: Expr) -> Expr:
    """Maximum value form vector field x"""
    return x.max()


def vec_min(x: Expr) -> Expr:
    """Minimum value form vector field x"""
    return x.min()


def vec_norm(x: Expr) -> Expr:
    """Sum of all absolute values of vector field x"""
    return x.abs().sum()


def vec_percentage(x: Expr, percentage: float = 0.5) -> Expr:
    """Percentile of vector field x"""
    return x.quantile(percentage)


def vec_powersum(x: Expr, constant: float = 2) -> Expr:
    """Sum of power of vector field x"""
    return (x ** constant).sum()


def vec_range(x: Expr) -> Expr:
    """Difference between maximum and minimum element in vector field x"""
    return x.max() - x.min()


def vec_skewness(x: Expr) -> Expr:
    """Skewness of vector field x"""
    return x.skew()


def vec_stddev(x: Expr) -> Expr:
    """Standard Deviation of vector field x"""
    return x.std(ddof=0)


def vec_sum(x: Expr) -> Expr:
    """Sum of vector field x"""
    return x.sum()
