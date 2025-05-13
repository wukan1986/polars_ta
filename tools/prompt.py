import inspect
from typing import List, Optional


def get_annotation(annotation):
    output = annotation.__name__
    if output == "Optional":
        output = str(annotation).split('.')[-1]
    return output


def get_parameter(p):
    annotation = get_annotation(p.annotation)
    # if annotation != "Expr":
    output = f"{p.name}:{annotation}"
    if p.default != inspect._empty:
        output += f"={p.default}"
    return output


def codegen_import_as(module: str,
                      include_modules: Optional[List[str]] = None,
                      include_func: Optional[List[str]] = None,
                      exclude_func: Optional[List[str]] = None,
                      include_parameter: Optional[List[str]] = None):
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
    include_parameter
        出现了同名int参数，添加前缀

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
    if include_parameter is None:
        include_parameter = []

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

        signature = inspect.signature(func)
        parameters = []
        for n, p in signature.parameters.items():
            if p.kind in (inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD):
                continue

            parameters.append(get_parameter(p))

        txts.append(f'- {name}({",".join(parameters)})->{get_annotation(signature.return_annotation)} : {func.__doc__.splitlines()[0]}')

    return txts


lines = []
lines += codegen_import_as('polars_ta.wq.arithmetic', exclude_func=['add', 'subtract', 'multiply', 'div', 'divide', 'reverse', 'inverse'])
lines += codegen_import_as('polars_ta.wq.time_series')
lines += codegen_import_as('polars_ta.wq.cross_sectional')
lines += codegen_import_as('polars_ta.wq.preprocess', exclude_func=['cs_mad_rank', 'cs_mad_rank2', 'cs_mad_rank2_resid', 'cs_mad_zscore', 'cs_mad_zscore_resid', 'cs_rank2'])
lines += codegen_import_as('polars_ta.wq.logical', include_func=['if_else', ])
text = '\n'.join(lines)
print(text)
