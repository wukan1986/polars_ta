# nan_to_null

It is important to handle nan and null values correctly. In many cases, null values may occur. Such as:

1. Stock suspension, data missing, etc.
2. Calculation exceptions, such as division by 0, log non-positive, etc.


## Null (None in Python) and NaN are different!

```python
None == None  # True # (It should be `None is None`)
np.nan == np.nan  # False
```

But in `pandas/numpy` many do not distinguish between `None` and `np.nan`, so you should always use `is_null/is_nan` functions to be safe.

1. In `pandas` 1.x, the representation of null values is:
  - floating point uses `nan`,
  - string uses `None`,
  - time uses `pd.NaT`,
  - integer has no null value representation, it can be converted to a floating-point number.
2. In `pandas` 2.x, the backend can use `Arrow`. It has two storage areas, one is the original data, and the other is the validity bitmap array, which marks whether it is `null`, so there is no data type restriction.
3. In `polars`, the backend is `Arrow`.

With a bitmap array, counting `null` values is faster than traversing the original array.

## `polars`

1. Most functions adapt to `null` but not `nan`, the best way should be using `fill_nan(None)` first.
2. To adapt to third-party packages, such as `TA-Lib`, when calling `to_numpy`, `null` will be automatically converted to `nan`, and the return value should be `pl.Series(, nan_to_null=True)`.
3. For the case of `null/nan` appearing in the middle, there is currently no good way to handle it.

## Reference

1. https://docs.pola.rs/user-guide/expressions/missing-data/#null-and-nan-values
2. https://pandas.pydata.org/docs/user_guide/missing_data.html





# nan_to_null

空值的处理是一件非常头疼的事性。多种情况下可能会出现空值。如：

1. 股票停牌、数据缺失 等
2. 计算异常，如：除0，log非正数 等

## None与NaN区别大不同

比较结果不同

```python
None == None  # True # 注意None is None才是规范的写法
np.nan == np.nan  # False
```

但在`pandas/numpy`中将`None`当成`np.nan`使用，这类容易混淆的地方，一定要使用`is_null/is_nan`函数才稳妥

1. 在pandas 1.x版本中，空值表示方法有：浮点用`nan`，字符串用`None`，时间用`pd.NaT`，整型没有空值表示方法，只能转成浮点
2. 在pandas 2.x版本中，后端可用`Arrow`。它有两块存储区域，一块还是原数据，另一块则是有效性位图数组，由它标记是否为`null`, 所以没有数据类型限制
3. 在polars中后端是`Arrow`

有位图数组，所以统计`null`数量肯定比遍历原数组要快

## polars

1. 大部分函数只适配了`null`，所以对于`nan`，最合适的处理方法是`fill_nan(None)`后再处理
2. 需要调用第三方函数时，以`TA-Lib`为例，`to_numpy`时会自动`null`转`nan`，返回值需`pl.Series(, nan_to_null=True)`
3. 对于中段出现`null/nan`的情况，目前还没有很好的处理方法

# 参考

1. https://docs.pola.rs/user-guide/expressions/missing-data/#null-and-nan-values
2. https://pandas.pydata.org/docs/user_guide/missing_data.html
