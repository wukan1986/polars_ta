# Code Conversion Tools

## TA-Lib tool (codegen_talib.py)

Wrap the original `TA-Lib` and save it to `polars_ta.talib.__init__.py`.

1. Support `Expr`
2. Support `skipna`
3. Multi-output support

## Prefix adding tool

Add the `ts_` prefix to some functions to facilitate use in `expr_codegen` and other tools.

1. `prefix_ta.py` adds a prefix to `polars_ta.ta` and saves it to `polars_ta.prefix.ta.py`
2. `prefix_tdx.py` adds a prefix to `polars_ta.tdx` and saves it to `polars_ta.prefix.tdx.py`
3. `prefix_talib.py` adds a prefix to `polars_ta.talib` and saves it to `polars_ta.prefix.talib.py` (same as `codegen_talib`)

# 代码转换工具

## TA-Lib工具(codegen_talib.py)

将原版`TA-Lib`封装，保存到`polars_ta.talib.__init__.py`，支持以下功能

1. 支持`Expr`
2. 支持跳过空值
3. 多输出适配

## 前缀添加工具

为部分函数添加`ts_`前缀，方便在`expr_codegen`等工具中使用

1. `prefix_ta.py`为`polars_ta.ta`添加前缀，保存到`polars_ta.prefix.ta.py`
2. `prefix_tdx.py`为`polars_ta.tdx`添加前缀，保存到`polars_ta.prefix.tdx.py`
3. `prefix_talib.py`为`polars_ta.talib`添加前缀，保存到`polars_ta.prefix.talib.py`(使用与codegen_talib同技术)



