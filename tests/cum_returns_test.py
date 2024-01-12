import time

import numpy as np
import pandas as pd

from polars_ta.performance.returns import cumulative_returns

_N = 250 * 10
_K = 5000

asset = [f's_{i}' for i in range(_K)]
date = pd.date_range('2015-1-1', periods=_N)

df = pd.DataFrame({
    'OPEN': np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0).reshape(-1),
    'HIGH': np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0).reshape(-1),
    'LOW': np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0).reshape(-1),
    'CLOSE': np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0).reshape(-1),
}, index=pd.MultiIndex.from_product([date, asset], names=['date', 'asset']))  # .reset_index()

a = df['CLOSE'].unstack()
b = cumulative_returns(a.pct_change(1), a > 0, 2)

t1 = time.perf_counter()
b = cumulative_returns(a.pct_change(1), a > 0, 20)
t2 = time.perf_counter()
print(t2 - t1)
