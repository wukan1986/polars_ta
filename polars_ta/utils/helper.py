"""
We follow the spirit of

https://github.com/pola-rs/polars/issues/9261

and adjusted it to fit our needs. The usage is as follows

expr.ta.func(..., skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False)

skip_nan: bool
    reduces speed
output_idx: int
    select a single column when outputing multiple columns
schema: list or tuple
    assign column names for multiple output columns
schema_format: str
    assign data types for multiple output columns
nan_to_null: bool
    replace all nan to null when outputing



以下是在github issues中 @cmdlineluser 提供的 注册命名空间 方案

https://github.com/pola-rs/polars/issues/9261

本人做了一定的调整。使用方法如下

expr.ta.func(..., skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False)

skip_nan: bool
    是否跳过空值。可以处理停牌无数据的问题，但会降低运行速度
output_idx: int
    多列输出时，选择只输出其中一列
schema: list or tuple
    返回为多列时，会组装成struct，可以提前设置子列的名字
schema_format: str
    为子列名指定格式
nan_to_null: bool
    返回值是否改成null

其它参数按**位置参数**和**命名参数**输入皆可
"""
import numpy as np
from polars import Expr, Float64, DataFrame, api
from polars import Series, Struct


def func_wrap_mn(func, cols,
                 *args,
                 skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False,
                 **kwargs):
    """multiple input multiple output, compatible with single input single output
    多输入多输出，兼容一输入一输出

    Parameters
    ----------
    func
    cols
    args
    skip_nan
    output_idx
    schema
    schema_format
    nan_to_null
    kwargs

    Returns
    -------
    Series

    """
    if cols.dtype.base_type() == Struct:
        # struct(['A', 'B']).ta.AROON
        _cols = [cols.struct[field] for field in cols.dtype.to_schema()]
    else:
        # col('A').ta.BBANDS
        _cols = [cols]

    if skip_nan:
        _cols = [c.cast(Float64).to_numpy() for c in _cols]
        _cols = np.vstack(_cols)

        # move nan to head
        idx1 = (~np.isnan(_cols).any(axis=0)).argsort(kind='stable')
        # restore nan
        idx2 = idx1.argsort(kind='stable')

        _cols = [_cols[i, idx1] for i in range(len(_cols))]
        result = func(*_cols, *args, **kwargs)

        if isinstance(result, tuple):
            result = tuple([_[idx2] for _ in result])
        else:
            result = result[idx2]
    else:
        result = func(*_cols, *args, **kwargs)

    if isinstance(result, tuple):
        if output_idx is None:
            # struct(['A').ta.BBANDS
            # you need alias outside, finally unnest
            if schema is None:
                schema = [f'column_{i}' for i in range(len(result))]

            schema = [schema_format.format(name) for name in schema]
            # nan_to_null is not effective for struct
            # nan_to_null 对struct中的nan无效
            return DataFrame(result, schema=schema, nan_to_null=nan_to_null).to_struct('')
        # output only one column
        if 0 <= output_idx < len(result):
            return Series(result[output_idx], nan_to_null=nan_to_null)
    elif not isinstance(result, Series):
        # col('A').bn.move_rank
        return Series(result, nan_to_null=nan_to_null)
    else:
        # col('A').ta.COS
        if nan_to_null:
            return result.fill_nan(None)
        else:
            return result


def func_wrap_11(func, cols,
                 *args,
                 skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False,
                 **kwargs):
    """single input single output is faster than multiple input multiple output
    一输入，一输出。处理速度能更快一些"""
    _cols = cols

    if skip_nan:
        _cols = _cols.cast(Float64).to_numpy()

        # move nan to head
        idx1 = (~np.isnan(_cols)).argsort(kind='stable')
        # restore nan
        idx2 = idx1.argsort(kind='stable')

        _cols = _cols[idx1]
        result = func(_cols, *args, **kwargs)

        result = result[idx2]
    else:
        result = func(_cols, *args, **kwargs)

    if isinstance(result, Series):
        if nan_to_null:
            return result.fill_nan(None)
        else:
            return result
    else:
        return Series(result, nan_to_null=nan_to_null)


class FuncHelper:
    def __init__(self, expr: Expr, lib=None, wrap=None) -> None:
        """initialization

        Parameters
        ----------
        expr
        lib:
            third-party packages required
        wrap:
            wrapper for third-party packages


        初始化

        Parameters
        ----------
        expr
        lib:
            需要调用的第三方库
        wrap:
            对第三方库的封装方法

        """
        object.__setattr__(self, '_expr', expr)
        object.__setattr__(self, '_lib', lib)
        object.__setattr__(self, '_wrap', wrap)

    def __getattribute__(self, name: str):
        _expr: Expr = object.__getattribute__(self, '_expr')
        _lib = object.__getattribute__(self, '_lib')
        _wrap = object.__getattribute__(self, '_wrap')
        _func = getattr(_lib, name)
        return (
            lambda *args, skip_nan=False, output_idx=None, schema=None, schema_format='{}', nan_to_null=False, **kwargs:
            _expr.map_batches(
                lambda cols: _wrap(_func, cols,
                                   *args,
                                   skip_nan=skip_nan, output_idx=output_idx, schema=schema, schema_format=schema_format, nan_to_null=nan_to_null,
                                   **kwargs)
            )
        )


@api.register_expr_namespace('ta')
class TaLibHelper(FuncHelper):
    def __init__(self, expr: Expr) -> None:
        import talib as ta
        super().__init__(expr, ta, func_wrap_mn)


@api.register_expr_namespace('bn')
class BottleneckHelper(FuncHelper):
    def __init__(self, expr: Expr) -> None:
        import bottleneck as bn
        super().__init__(expr, bn, func_wrap_11)
