import inspect
from typing import List, Optional


def codegen_import_as(module: str, prefix: str = 'ts_',
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
    prefix
        需要添加的前缀
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
    txts = []
    for name, func in funcs:
        if func.__module__ not in include_modules:
            continue
        if name in exclude_func:
            continue

        add_prefix = False
        if name.startswith(prefix):
            add_prefix = False
        elif name in include_func:
            add_prefix = True
        else:
            p = inspect.signature(func).parameters
            for n in include_parameter:
                if n in p:
                    if p[n].annotation == int:
                        add_prefix = True
                    break

        if add_prefix:
            txts.append(f'from {module} import {name} as {prefix}{name}  # noqa')
        else:
            txts.append(f'from {module} import {name}  # noqa')

    return txts


def save(txts, module, write=False):
    m = __import__(module, fromlist=['*'])
    file = m.__file__
    print('save to', file)
    text = '\n'.join(txts)
    if write:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        print(text)
