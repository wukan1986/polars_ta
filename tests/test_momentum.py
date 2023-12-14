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

    def test_WILLR(self):
        from polars_ta.ta.momentum import WILLR

        result1 = talib.WILLR(self.high_np, self.low_np, self.close_np, timeperiod=5)
        result2 = self.df_pl.select(WILLR(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=5))
        # !!! 注意，与talib版的区别
        result3 = result2['high'].to_numpy() * (-100)

        assert np.allclose(result1, result3, equal_nan=True)

    def test_STOCHF_fastk(self):
        from polars_ta.ta.momentum import STOCHF_fastk

        result1, _ = talib.STOCHF(self.high_np, self.low_np, self.close_np, fastk_period=5, fastd_period=1, fastd_matype=0)
        result2 = self.df_pl.select(STOCHF_fastk(pl.col("high"), pl.col("low"), pl.col("close"), fastk_period=5))
        # !!! 注意，与talib版的区别
        result3 = result2['close'].to_numpy() * 100
        # print()
        # print(result1)
        # print(result3)

        assert np.allclose(result1, result3, equal_nan=True)

    def test_STOCHF_fastd(self):
        from polars_ta.ta.momentum import STOCHF_fastd

        _, result1 = talib.STOCHF(self.high_np, self.low_np, self.close_np, fastk_period=5, fastd_period=3, fastd_matype=0)
        result2 = self.df_pl.select(STOCHF_fastd(pl.col("high"), pl.col("low"), pl.col("close"), fastk_period=5, fastd_period=3))
        # !!! 注意，与talib版的区别
        result3 = result2['close'].to_numpy() * 100
        # print()
        # print(result1)
        # print(result3)

        assert np.allclose(result1, result3, equal_nan=True)

    def test_MACD(self):
        talib.set_compatibility(1)

        from polars_ta.ta.momentum import MACD_macd

        fastperiod = 3
        slowperiod = 5

        result1, _, _ = talib.MACD(self.close_np, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=1)
        result2 = self.df_pl.select(MACD_macd(pl.col("close"), fastperiod=fastperiod, slowperiod=slowperiod))
        result3 = result2['close'].to_numpy()
        # print()
        # print(result1)
        # print(result3)

        assert np.allclose(result1[slowperiod:], result3[slowperiod:], equal_nan=True)

    def test_TRIX(self):
        talib.set_compatibility(1)

        from polars_ta.ta.momentum import TRIX

        result1 = talib.TRIX(self.close_np, timeperiod=30)
        result2 = self.df_pl.select(TRIX(pl.col("close"), timeperiod=30))
        result3 = result2['close'].to_numpy() * 100
        # print()
        # print(result1)
        # print(result3)

        assert np.allclose(result1, result3, equal_nan=True)
