import numpy as np
import polars as pl

from polars_ta.wq.arithmetic import add


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

    def test_add(self):
        print(self.df_pl.schema)
        result2 = self.df_pl.select(add(pl.col('high'), pl.col('low'), pl.col('null'), filter_=True))
        print(result2)
        # result3 = self.df_pl.select(add(pl.col('high'), pl.col('low'), pl.col('literal'), filter_=True))
        # print(result3)

    def test_fraction(self):
        from polars_ta.wq.arithmetic import fraction

        a = np.array([5.63, -4.59])
        df = pl.DataFrame({'a': a})
        result1 = df.select(fraction(pl.col('a')))
        result3 = np.array([0.63, -0.59])
        assert np.allclose(result1.to_numpy()[:, 0], result3, equal_nan=True)
