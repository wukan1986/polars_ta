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
        self.volume_np = np.arange(100, dtype=float) + np.random.rand(100)

        self.df_pl = pl.DataFrame([self.high_np, self.low_np, self.close_np, self.volume_np],
                                  schema=["high", "low", "close", "volume"])

    def test_AD(self):
        from polars_ta.ta.volume import AD

        result1 = talib.AD(self.high_np, self.low_np, self.close_np, self.volume_np)
        result2 = self.df_pl.select(AD(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")))
        result3 = result2['close'].to_numpy()
        # print()
        # print(result1)
        # print(result3)

        assert np.allclose(result1, result3, equal_nan=True)

    def test_ADOSC(self):
        from polars_ta.ta.volume import ADOSC

        result1 = talib.ADOSC(self.high_np, self.low_np, self.close_np, self.volume_np)
        result2 = self.df_pl.select(ADOSC(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")))
        result3 = result2['close'].to_numpy()
        # print()
        # print(result1)
        # print(result3)

        assert np.allclose(result1, result3, equal_nan=True)

    def test_OBV(self):
        from polars_ta.ta.volume import OBV

        result1 = talib.OBV(self.close_np, self.volume_np)
        result2 = self.df_pl.select(OBV(pl.col("close"), pl.col("volume")))
        result3 = result2['close'].to_numpy()
        # print()
        # print(result1)
        # print(result3)

        assert np.allclose(result1, result3, equal_nan=True)
