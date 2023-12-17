"""
以下是polars提供的方案基础上封装的方案
func(expr)

"""

from polars_ta.talib import *

df = pl.DataFrame(
    {
        "A": [5, None, 3, 2, 1],
        "B": [5, 4, None, 2, 1],
        "C": [5, 4, 3, 2, 1],
    }
)

df = df.with_columns([
    # # 一输入一输出，不需处理空值
    COS(pl.col('A')).alias('COS'),
    #
    # 多输入一输出
    ATR(pl.col('A'), pl.col('B'), pl.col('C'), 2, nan_to_null=True).alias('ATR2'),
    #
    # 一输入多输出，可通过prefix为多输出添加前缀
    BBANDS(pl.col('A'), timeperiod=2, skip_nan=True, schema_format='bbands_{}').alias('BBANDS'),

    # 多输入多输出。可通过schema直接添加
    AROON('A', 'B', timeperiod=2, skip_nan=True, schema=('aroondown', 'aroonup'), nan_to_null=True).alias('AROON'),

])

print(df)

df = df.unnest('BBANDS', 'AROON')
print(df)
