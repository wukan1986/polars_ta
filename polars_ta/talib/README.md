# polars_ta.talib

Inside this package, files are generated by `tools.codegen_talib2`.
It is a wrapper of `talib` functions, with the following features:

1. Input and output are `Expr` instead of `Series`
2. ~~Add skipna feature (not efficient, will update when `polars` support backward fill)~~

本包由`tools.codegen_talib2`自动生成，是对`talib`代码的封装。实现了以下功能
1. 输入输出由`Series`改`Expr`
2. ~~添加跳过空值功能（效率不高，等`polars`支持反向填充，此部分将更新）~~

