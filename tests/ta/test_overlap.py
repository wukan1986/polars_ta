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

    def test_SMA(self):
        from polars_ta.ta.overlap import SMA

        result1 = talib.SMA(self.close_np, timeperiod=3)
        result2 = self.df_pl.select(SMA(pl.col("close"), d=3))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)

    def test_WMA(self):
        from polars_ta.ta.overlap import WMA

        result1 = talib.WMA(self.close_np, timeperiod=15)
        result2 = self.df_pl.select(WMA(pl.col("close"), d=15))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)

    def test_EMA(self):
        from polars_ta.ta.overlap import EMA
        # !!! 此处非常重要，有部分函数受此影响
        # https://github.com/TA-Lib/ta-lib-python/blob/master/talib/_ta_lib.pxd#L28
        talib.set_compatibility(1)

        result1 = talib.EMA(self.close_np, timeperiod=5)
        result2 = self.df_pl.select(EMA(pl.col("close"), timeperiod=5))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)

    def test_DEMA(self):
        from polars_ta.ta.overlap import DEMA
        # !!! 此处非常重要，有部分函数受此影响
        talib.set_compatibility(1)

        result1 = talib.DEMA(self.close_np, timeperiod=3)
        result2 = self.df_pl.select(DEMA(pl.col("close"), timeperiod=3))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)

    def test_TEMA(self):
        from polars_ta.ta.overlap import TEMA
        # !!! 此处非常重要，有部分函数受此影响
        talib.set_compatibility(1)

        result1 = talib.TEMA(self.close_np, timeperiod=3)
        result2 = self.df_pl.select(TEMA(pl.col("close"), timeperiod=3))
        result3 = result2['close'].to_numpy()
        # print()
        # print(result1)
        # print(result3)

        assert np.allclose(result1, result3, equal_nan=True)

    def test_TRIMA(self):
        from polars_ta.ta.overlap import TRIMA

        result1 = talib.TRIMA(self.close_np, timeperiod=6)
        result2 = self.df_pl.select(TRIMA(pl.col("close"), timeperiod=6))
        result3 = result2['close'].to_numpy()
        # print()
        # print(result1)
        # print(result3)

        assert np.allclose(result1, result3, equal_nan=True)

    # def test_KAMA(self):
    #     # !!! 此处非常重要，有部分函数受此影响
    #     talib.set_compatibility(1)
    #
    #     from polars_ta.talib.overlap import KAMA
    #
    #     result1 = talib.KAMA(self.close_np, timeperiod=5)
    #     result2 = self.df_pl.select(KAMA(pl.col("close"), timeperiod=5))
    #     result3 = result2['close'].to_numpy()
    #     # print()
    #     # print(result1)
    #     # print(result3)
    #
    #     assert np.allclose(result1, result3, equal_nan=True)

    def test_BBANDS(self):
        from polars_ta.ta.overlap import BBANDS

        result1, _, _ = talib.BBANDS(self.close_np, timeperiod=3)
        result2 = self.df_pl.select(BBANDS(pl.col("close"), timeperiod=3).struct[0])
        result3 = result2['upperband'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)

    def test_MIDPOINT(self):
        from polars_ta.ta.overlap import MIDPOINT

        result1 = talib.MIDPOINT(self.close_np, timeperiod=3)
        result2 = self.df_pl.select(MIDPOINT(pl.col("close"), timeperiod=3))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)

    def test_MIDPRICE(self):
        from polars_ta.ta.overlap import MIDPRICE

        result1 = talib.MIDPRICE(self.high_np, self.low_np, timeperiod=3)
        result2 = self.df_pl.select(MIDPRICE(pl.col("high"), pl.col("low"), timeperiod=3))
        result3 = result2['high'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)
