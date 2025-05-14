import inspect
from typing import List, Optional


def get_annotation(annotation):
    output = annotation.__name__
    if output == "Optional":
        output = str(annotation).split('.')[-1]
    return output


def get_parameter(p):
    annotation = get_annotation(p.annotation)
    if p.kind == inspect._ParameterKind.VAR_POSITIONAL:
        output = f"*{p.name}"
    elif p.kind == inspect._ParameterKind.VAR_KEYWORD:
        output = f"**{p.name}"
    else:
        output = p.name
    if annotation not in ("Expr", "_empty"):
        output += f":{annotation}"
    if p.default != inspect._empty:
        output += f"={p.default}"

    return output


def codegen_import_as(module: str,
                      include_modules: Optional[List[str]] = None,
                      include_func: Optional[List[str]] = None,
                      exclude_func: Optional[List[str]] = None):
    """Generate codes by `reflection`
    通过反射，生成代码的小工具

    Parameters
    ----------
    module
        模块全名
    include_modules
        通过`from import`导入的函数也参与判断
    include_func
        指定的函数名不做检查，直接添加前缀
    exclude_func
        指定的函数名直接跳过不导入

    Notes
    -----

    """
    if include_modules is None:
        include_modules = [module]
    else:
        include_modules += [module]
    if include_func is None:
        include_func = []
    if exclude_func is None:
        exclude_func = []

    m = __import__(module, fromlist=['*'])
    funcs = inspect.getmembers(m, inspect.isfunction)
    funcs = [f for f in funcs if not f[0].startswith('_')]
    txts = [f"### {module}"]
    for name, func in funcs:
        if func.__module__ not in include_modules:
            continue
        if exclude_func and name in exclude_func:
            continue
        if include_func and name not in include_func:
            continue

        # if name != "cs_resid":
        #     continue

        signature = inspect.signature(func)
        parameters = []
        for n, p in signature.parameters.items():
            if p.name == "min_samples":
                continue
            parameters.append(get_parameter(p))

        # txts.append(f'- {name}({",".join(parameters)})->{get_annotation(signature.return_annotation)} : {func.__doc__.splitlines()[0]}')
        txts.append(f'- {name}({",".join(parameters)}) : {func.__doc__.splitlines()[0]}')

    return txts


# 过滤一些很少用到的函数，最好控制在5000字内（百度5000字限制）
lines = []
lines += codegen_import_as('polars_ta.wq.arithmetic',
                           exclude_func=['add', 'subtract', 'multiply', 'div', 'divide', 'reverse', 'inverse', "mean",
                                         'arc_cos', 'arc_sin', 'arc_tan', 'arc_tan2', 'cot', 'cosh', 'tanh', 'sinh', 'degrees', 'radians',
                                         's_log_1p', 'softsign', 'log2',
                                         ])
lines += codegen_import_as('polars_ta.wq.time_series',
                           exclude_func=['ts_co_kurtosis', 'ts_co_skewness', 'ts_count_nans', 'ts_count_nulls',
                                         'ts_cum_prod', 'ts_cum_prod_by', 'ts_cum_sum', 'ts_cum_sum_by', 'ts_cum_sum_reset',
                                         "ts_min_max_cps", "ts_min_max_diff", "ts_signals_to_size", 'ts_sum_split_by'])
lines += codegen_import_as('polars_ta.wq.cross_sectional',
                           exclude_func=['cs_fill_except_all_null', 'cs_fill_mean', 'cs_fill_null', 'cs_top_bottom', 'cs_one_side'])
lines += codegen_import_as('polars_ta.wq.preprocess',
                           exclude_func=['cs_mad_rank', 'cs_mad_rank2', 'cs_mad_rank2_resid', 'cs_mad_zscore', 'cs_mad_zscore_resid', 'cs_rank2'])
lines += codegen_import_as('polars_ta.wq.logical',
                           include_func=['if_else', ])
lines += codegen_import_as('polars_ta.wq.transformational',
                           include_func=['cut', 'sigmoid', 'bool_', 'int_', 'float_'])
text = '\n'.join(lines)
print(text)
