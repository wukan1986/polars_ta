import time

import numpy as np
import polars as pl
from numba import jit

from polars_ta.utils.numba_ import nb_roll_sum, batches_1, roll_sum, roll_cov


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def nb_sum(x):
    return np.sum(x)


df = pl.DataFrame({'A': range(100000), 'B': range(100000)})
a = df.with_columns([
    pl.col('A').rolling_sum(10).alias('a1'),
    pl.col('A').rolling_map(lambda x: x.sum(), 10).alias('a2'),
    pl.col('A').rolling_map(lambda x: nb_sum(x.to_numpy()), 10).alias('a3'),
    roll_sum(pl.col('A'), 10).alias('a4'),
    pl.col('A').map_batches(lambda x: batches_1(x, 10, nb_roll_sum)).alias('a5'),
    pl.rolling_cov(pl.col('A'), pl.col('B'), window_size=10).alias('a6'),
    roll_cov(pl.col('A'), pl.col('B'), 10).alias('a7'),
])
print(a)

t1 = time.perf_counter()
for i in range(10):
    a = df.with_columns([
        pl.col('A').rolling_sum(10).alias('a1'),
    ])
t2 = time.perf_counter()
print(t2 - t1)

t1 = time.perf_counter()
for i in range(10):
    a = df.with_columns([
        pl.col('A').rolling_map(lambda x: x.sum(), 10).alias('a2'),
    ])
t2 = time.perf_counter()
print(t2 - t1)

t1 = time.perf_counter()
for i in range(10):
    a = df.with_columns([
        pl.col('A').rolling_map(lambda x: nb_sum(x.to_numpy()), 10).alias('a3'),
    ])
t2 = time.perf_counter()
print(t2 - t1)

t1 = time.perf_counter()
for i in range(10):
    a = df.with_columns([
        pl.col('A').map_batches(lambda x: pl.Series(nb_roll_sum(x.to_numpy(), 10), nan_to_null=True)).alias('a4'),
    ])
t2 = time.perf_counter()
print(t2 - t1)

t1 = time.perf_counter()
for i in range(10):
    a = df.with_columns([
        pl.col('A').map_batches(lambda x: batches_1(x, 10, nb_roll_sum)).alias('a5'),
    ])
t2 = time.perf_counter()
print(t2 - t1)
