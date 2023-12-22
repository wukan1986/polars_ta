# polars_ta.talib

本包由`tools.codegen_talib`自动生成，是对`talib`代码的封装。实现了以下功能
1. 输入输出由`Series`改`Expr`
2. 添加跳过空值功能（效率不高，等`polars`支持反向填充，此部分将更新）

