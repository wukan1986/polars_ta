# polars_ta.ta

1. 本文件夹中模仿`talib`，实现同名函数的`polars`版
2. 由于减少了python与c来回调用，理论上比直接调用`talib`快
3. 同功能函数优先从`wq`中导入，没有了才考虑是否实现