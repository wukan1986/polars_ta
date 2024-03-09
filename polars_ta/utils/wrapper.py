"""
Another wrapper for raised github issue

before:
expr.ta.func(..., skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False)

now:
func(expr, ..., skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False)

This wrapper allows the prefix expression to be easily used in genetic algorithm tools

此处是对github issues上提出的 注册命名空间 方案的再封装

之前的使用方法为
expr.ta.func(..., skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False)

封装后方法为
func(expr, ..., skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False)

此种封装方法的优点是前缀表达式方便输入到遗传算法工具包中使用
"""

from functools import wraps

import talib as _talib
from polars import Expr
from polars import struct
from talib import abstract as _abstract

from polars_ta.utils.helper import TaLibHelper

_ = TaLibHelper


def ta_func(func, func_name, input_names, output_names,
            *args,
            skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False,
            **kwargs):
    """

    Parameters
    ----------
    func
    func_name
    input_names: list of str
        函数输入名
    output_names: list of str
        函数输出名
    args
        位置参数
    skip_nan
    output_idx
    schema
    schema_format
    nan_to_null
    kwargs
        命名参数

    Returns
    -------

    """
    exprs = [arg for arg in args if isinstance(arg, Expr)]
    param = [arg for arg in args if not isinstance(arg, Expr)]

    if len(exprs) == 1:
        ef = getattr(exprs[0].ta, func_name)
    else:
        ef = getattr(struct(*exprs).ta, func_name)

    return ef(*param,
              skip_nan=skip_nan, output_idx=output_idx, schema=schema, schema_format=schema_format, nan_to_null=nan_to_null,
              **kwargs)


def ta_decorator(func, func_name, input_names, output_names):
    @wraps(func)
    def decorated(*args,
                  skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False,
                  **kwargs):
        return ta_func(func, func_name, input_names, output_names,
                       *args,
                       skip_nan=skip_nan, output_idx=output_idx, schema=schema, schema_format=schema_format, nan_to_null=nan_to_null,
                       **kwargs)

    return decorated


def init(to_globals=False, name_format='{}'):
    """初始化环境

    Parameters
    ----------
    to_globals: bool
        是否注册到全局变量中
    name_format: str
        函数名格式

    Returns
    -------
    class
        对象实例，可调用其内的方法

    """

    class TA_LIB:
        pass

    lib = TA_LIB()
    for i, func_name in enumerate(_talib.get_functions()):
        """talib遍历"""
        _ta_func = getattr(_talib, func_name)
        info = _abstract.Function(func_name).info
        output_names = info['output_names']
        input_names = info['input_names']

        # 创建函数
        f = ta_decorator(_ta_func, func_name, input_names, output_names)

        name = name_format.format(func_name)
        setattr(lib, name, f)

        if to_globals:
            from inspect import currentframe
            frame = currentframe().f_back
            frame.f_globals[name] = f

    return lib
