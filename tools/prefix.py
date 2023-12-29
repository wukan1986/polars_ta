import inspect


def codegen_import(module, prefix='ts_', include_modules=[], include_func=[], include_parameter=[]):
    """通过反射，生成代码的小工具"""
    m = __import__(module, fromlist=['*'])
    funcs = inspect.getmembers(m, inspect.isfunction)
    funcs = [f for f in funcs if not f[0].startswith('_')]
    txts = []
    for name, func in funcs:
        if len(include_modules) == 0:
            if func.__module__ != module:
                continue
        else:
            if func.__module__ not in include_modules:
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
