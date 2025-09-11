import numpy as np
import polars as pl
import talib


class TestDemoClass:
    high_np = None
    low_np = None
    close_np = None
    df_pl = None

    def setup_class(self):
        self.high_np = 0 + np.random.rand(100)
        self.low_np = 0 - np.random.rand(100)
        self.close_np = np.arange(100, dtype=float)+ np.random.rand(100)

        self.df_pl = pl.DataFrame([self.high_np, self.low_np, self.close_np],
                                  schema=["high", "low", "close"])

    def test_WILLR(self):
        from polars_ta.ta.momentum import WILLR

        result1 = talib.WILLR(self.high_np, self.low_np, self.close_np, timeperiod=5)
        result2 = self.df_pl.select(WILLR(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=5))
        # !!! 注意，与talib版的区别
        result3 = result2['high'].to_numpy() * (-100)

        print()
        print(result1)
        print(result3)

        # assert np.allclose(result1, result3, equal_nan=True)


    def test_STOCHF_fastd(self):
        from polars_ta.ta.momentum import STOCHF

        _, result1 = talib.STOCHF(self.high_np, self.low_np, self.close_np, fastk_period=5, fastd_period=3, fastd_matype=0)
        result2 = self.df_pl.select(STOCHF(pl.col("high"), pl.col("low"), pl.col("close"), fastk_period=5, fastd_period=3).struct[1])
        # !!! 注意，与talib版的区别
        result3 = result2['fastd'].to_numpy() * 100
        print()
        print(result1)
        print(result3)

        # assert np.allclose(result1, result3, equal_nan=True)

    def test_MACD(self):
        talib.set_compatibility(1)

        from polars_ta.ta.momentum import MACD

        fastperiod = 3
        slowperiod = 5

        result1, _, _ = talib.MACD(self.close_np, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=1)
        result2 = self.df_pl.select(MACD(pl.col("close"), fastperiod=fastperiod, slowperiod=slowperiod).struct[0])
        result3 = result2['macd'].to_numpy()
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

    def test_AROON(self):
        timeperiod = 10

        from polars_ta.ta.momentum import AROON

        _, result1 = talib.AROON(self.high_np, self.low_np, timeperiod=timeperiod)
        result2 = self.df_pl.select(AROON(pl.col("high"), pl.col("low"), timeperiod=timeperiod).alias('high').struct[1])
        result3 = result2['aroonup'].to_numpy() * 100
        print()
        print(result1)
        print(result3)

        # assert np.allclose(result1[timeperiod:], result3[timeperiod:], equal_nan=True)

    def test_RSI(self):
        timeperiod = 10

        from polars_ta.ta.momentum import RSI

        result1 = talib.RSI(self.close_np, timeperiod=timeperiod)
        result2 = self.df_pl.select(RSI(pl.col("close"), timeperiod=timeperiod))
        result3 = result2['close'].to_numpy() * 100
        # print()
        # print(result1)
        # print(result3)
        assert np.allclose(result1[timeperiod:], result3[timeperiod:], equal_nan=True)