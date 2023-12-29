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



