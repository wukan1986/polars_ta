# polars_ta.tdx

1. Follows the `tdx` naming convention
2. Except for some element-wise functions, all functions are time-series functions. Pay special attention to `MAX` and similar functions.
3. First import from `tdx`, then from `wq`, and finally from `ta`. Only implement the function if it is not available.


1. 函数名称按照通达信来
2. 除了部分按元素计算的函数，其它都为时序函数，特别注意`MAX`等一类不要混淆
3. 优先从`tdx`中导入，然后从`wq`中导入，最后从`ta`，没有的才实现