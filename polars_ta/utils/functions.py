import inspect
from functools import wraps
from inspect import currentframe

import polars as pl

import polars_ta


def const_to_expr(func):
    """将表达式中不合法的常量,改成表达式"""

    def repeat_const(a, p):
        if p.annotation.__name__ == "Expr" and not isinstance(a, pl.Expr):
            return pl.repeat(a, pl.len())
        else:
            return a

    @wraps(func)
    def decorated(*args):
        parameters = inspect.signature(func).parameters
        args = [repeat_const(a, p) for (n, p), a in zip(parameters.items(), args)]
        return func(*args)

    return decorated


def apply_const_to_expr():
    """应用常量转表达式功能到所有函数

    对于不合法的表达式参数，可以一定程度上兼容，适用于遗传算法中表达式简化成常量的情况
    """
    frame = currentframe().f_back
    for k, v in frame.f_globals.items():
        if inspect.isfunction(v) and v.__module__ and v.__module__.startswith(polars_ta.__package__):
            frame.f_locals[k] = const_to_expr(v)
