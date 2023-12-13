"""
以下是polars提供的实现调用第三方库的方案
expr.ta.func

"""
import polars as pl

from polars_ta.utils.helper import TaLibHelper

_ = TaLibHelper

df = pl.DataFrame(
    {
        "A": [5, None, 3, 2, 1],
        "B": [5, 4, None, 2, 1],
        "C": [5, 4, 3, 2, 1],
    }
)

df = df.with_columns([
    # 一输入一输出，不需处理空值
    pl.col('A').ta.COS().alias('COS'),
    # 一输入多输出
    pl.col('A').ta.BBANDS(timeperiod=2, schema=['upperband', 'middleband', 'lowerband'], skip_nan=True).alias('BBANDS'),
    # 多输入一输出
    pl.struct(['A', 'B', 'C']).ta.ATR(timeperiod=2, skip_nan=True).alias('ATR'),
    # 多输入多输出
    pl.struct(['A', 'B']).ta.AROON(timeperiod=2, schema=('aroondown', 'aroonup'), skip_nan=True).alias('AROON'),
    # 多输入一输出
    pl.struct(['A', 'B']).ta.AROON(timeperiod=2, skip_nan=True, output_idx=1).alias('aroonup1'),
    # 调用另一库
    pl.col('A').bn.move_rank(window=2, skip_nan=False).alias('move_rank'),
])

print(df)

df = df.unnest('BBANDS', 'AROON')
print(df)
