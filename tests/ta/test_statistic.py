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
        self.close_np = np.arange(100, dtype=float)

        self.df_pl = pl.DataFrame([self.high_np, self.low_np, self.close_np],
                                  schema=["high", "low", "close"])
        self.df_pd = self.df_pl.to_pandas()

    def test_STDDEV(self):
        from polars_ta.ta.statistic import STDDEV

        result1 = talib.STDDEV(self.close_np, timeperiod=3)
        result2 = self.df_pl.select(STDDEV(pl.col("close"), timeperiod=3))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)

    def test_VAR(self):
        from polars_ta.ta.statistic import VAR

        result1 = talib.VAR(self.close_np, timeperiod=5)
        result2 = self.df_pl.select(VAR(pl.col("close"), timeperiod=5))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)

    def test_CORREL(self):
        from polars_ta.ta.statistic import CORREL

        result1 = talib.CORREL(self.close_np, self.high_np, timeperiod=5)
        result2 = self.df_pl.select(CORREL(pl.col("close"), pl.col('high'), timeperiod=5))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)
