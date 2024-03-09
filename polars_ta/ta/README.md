# polars_ta.ta

1. Files in this folder mimic `talib`, and implement `polars` versions for the same functions
2. Since we reduce the functino calls between `Python` and `C` code, it should be faster than `talib`.
3. We first try to import from `ta`, then from `wq`, and only implement the function if it is not available.
4. When there is a circular dependency, we use `polars` instead.


1. 本文件夹中模仿`talib`，实现同名函数的`polars`版
2. 由于减少了python与c来回调用，理论上比直接调用`talib`快
3. 优先从`ta`中导入，然后从`wq`中导入，没有的才实现
4. 出现循环依赖时，使用`polars`