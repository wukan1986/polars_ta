import numpy as np
import polars as pl
import talib


class TestDemoClass:
    high_np = None
    low_np = None
    close_np = None
    df_pl = None

    def setup_class(self):
        self.high_np = np.arange(100, dtype=float) + np.random.rand(100)
        self.low_np = np.arange(100, dtype=float) - np.random.rand(100)
        self.close_np = np.random.rand(100)

        self.df_pl = pl.DataFrame([self.high_np, self.low_np, self.close_np],
                                  schema=["high", "low", "close"])

    def test_MAX(self):
        from polars_ta.ta.operators import MAX

        result1 = talib.MAX(self.close_np, timeperiod=10)
        result2 = self.df_pl.select(MAX(pl.col("close"), timeperiod=10))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)

    def test_MIN(self):
        from polars_ta.ta.operators import MIN

        result1 = talib.MIN(self.close_np, timeperiod=10)
        result2 = self.df_pl.select(MIN(pl.col("close"), timeperiod=10))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)

    def test_MAXINDEX(self):
        from polars_ta.ta.operators import MAXINDEX

        result1 = talib.MAXINDEX(self.close_np, timeperiod=5)
        result2 = self.df_pl.select(MAXINDEX(pl.col("close"), timeperiod=5))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1[5:], result3[5:], equal_nan=True)
