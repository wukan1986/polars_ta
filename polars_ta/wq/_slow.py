"""
Algorithm in this file is slow, and has been replaced by numba and other methods
本目录下算法有些慢，已经用numba等其它方法代替
"""
import numpy as np
from polars import Series, Expr


def _arg_max(x: Series):
    """
    Notes
    -----
    TODO 等polars推出rolling_arg_max(reverse=True)这个问题能好转

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


def ts_product(x: Expr, d: int = 5) -> Expr:
    return x.rolling_map(np.nanprod, d)
