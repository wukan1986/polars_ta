"""
以下是polars提供的方案基础上封装的方案
func(expr)

"""
import polars as pl

from polars_ta.utils.wrapper import init

df = pl.DataFrame(
    {
        "A": [5, None, 3, 2, 1],
        "B": [5, 4, None, 2, 1],
        "C": [5, 4, 3, 2, 1],
    }
)

# 已经注册到全局，可直接使用。但IDE中没有智能提示
init(to_globals=True, name_format='{}')

df = df.with_columns([
    # 一输入一输出，不需处理空值
    COS(pl.col('A')).alias('COS'),

    # 多输入一输出
    ATR(pl.struct(['A', 'B', 'C']), timeperiod=2).alias('ATR1'),
    ATR(pl.col('A'), pl.col('B'), pl.col('C'), 2, skip_nan=True).alias('ATR2'),

    # 一输入多输出，可通过prefix为多输出添加前缀
    BBANDS(pl.col('A'), timeperiod=2, skip_nan=True, schema_format='bbands_{}').alias('BBANDS'),

    # 多输入多输出。可通过schema直接添加
    AROON(pl.struct(['A', 'B']), timeperiod=2, skip_nan=True, schema=('aroondown', 'aroonup')).alias('AROON'),

])

print(df)

df = df.unnest('BBANDS', 'AROON')
print(df)

# 另一种调用方法
t = init(to_globals=False, name_format='ts_{}')
df = df.with_columns([
    # 一输入一输出，不需处理空值
    t.ts_COS(pl.col('A')).alias('ts_COS'),
])

print(df)
