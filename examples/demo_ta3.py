"""
This is how we wrap upon polars's code
以下是polars提供的方案基础上封装的方案
func(expr)

"""
import polars as pl

from polars_ta.talib import *

df = pl.DataFrame(
    {
        "A": [5, None, 3, 2, 1],
        "B": [5, 4, None, 2, 1],
        "C": [5, 4, 3, 2, 1],
    }
)

df = df.with_columns([
    # single input single ouput, no need to handle null/nan values
    # # 一输入一输出，不需处理空值
    COS(pl.col('A')).alias('COS'),
    # 多输入一输出
    ATR(pl.col('A'), pl.col('B'), pl.col('C'), 2).alias('ATR2'),
    # 原为一输入多输出，通过ret_idx指定一输出
    BBANDS(pl.col('A'), timeperiod=2, ret_idx=1).alias('BBANDS'),
    # 原为多输入多输出，通过ret_idx指定一输出
    AROON('A', 'B', timeperiod=2).alias('AROON'),

])

print(df)
