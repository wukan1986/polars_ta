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
        from polars_ta.tdx.statistic import AVEDEV

        df = pl.DataFrame({'A': range(100)})
        print(df)
        result1 = self.df_pl.select(AVEDEV(pl.col('high'), 10))
        print(result1)
