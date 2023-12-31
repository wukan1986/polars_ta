import time

import numpy as np
import polars as pl


class TestDemoClass:
    high_np = None
    low_np = None
    close_np = None
    df_pl = None

    def setup_class(self):
        self.high_np = np.arange(100, dtype=float) + np.random.rand(100)
        self.low_np = np.arange(100, dtype=float) - np.random.rand(100)
        self.close_np = np.arange(100, dtype=float)

        self.df_pl = pl.DataFrame([self.high_np, self.low_np, self.close_np],
                                  schema=["high", "low", "close"])
        self.df_pl = self.df_pl.with_columns(pl.lit(None).alias("null"))

    def test_AVEDEV(self):
        from polars_ta.tdx._slow import AVEDEV as func_slow
        from polars_ta.tdx.statistic import AVEDEV as func_fast

        xx = self.df_pl.with_columns(
            a1=func_slow(pl.col('high'), 10),
            a2=func_fast(pl.col('high'), 10),
        )
        print(xx)

        t1 = time.perf_counter()
        for i in range(1000):
            result2 = self.df_pl.with_columns(
                a1=func_slow(pl.col('high'), 10),
            )
        t2 = time.perf_counter()
        print(t2 - t1)

        t1 = time.perf_counter()
        for i in range(1000):
            result2 = self.df_pl.with_columns(
                a1=func_fast(pl.col('high'), 10),
            )
        t2 = time.perf_counter()
        print(t2 - t1)


