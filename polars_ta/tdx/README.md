# polars_ta.tdx

1. 函数名称按照通达信来
2. 除了部分按元素计算的函数，其它都为时序函数，特别注意`MAX`等一类不要混淆
3. 优先从`tdx`中导入，然后从`wq`中导入，最后从`tq`，没有的才实现