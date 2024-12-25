import numpy as np
import polars as pl

from polars_ta.tdx.pattern import ts_WINNER_COST

high = np.array([10.4, 10.2, 10.2, 10.4, 10.5, 10.7, 10.7, 10.7, 10.8, 10.9])
low = np.array([10.3, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9])
avg = np.array([10.3, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9])
close = np.array([10.3, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9])
turnover = np.array([0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
cost = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

step = 0.1

df = pl.DataFrame({'high': high, 'low': low, 'avg': avg, 'close': close, 'turnover': turnover, 'cost': cost})
df = df.with_columns(WINNER=ts_WINNER_COST(pl.col('high'), pl.col('low'), pl.col('avg'), pl.col('turnover'), pl.col('close'), 0.5, step=step))
print(df)
