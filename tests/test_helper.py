import numpy as np
import polars as pl
import talib

from polars_ta.utils.helper import TaLibHelper

_ = TaLibHelper


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
        result1 = talib.SMA(self.close_np, timeperiod=5)
        result2 = self.df_pl.select(pl.col('close').ta.SMA(timeperiod=5))
        result3 = result2['close'].to_numpy()
        # print()
        # print(result1)
        # print(result3)

        assert np.allclose(result1, result3, equal_nan=True)

    def test_BBANDS(self):
        result1, _, _ = talib.BBANDS(self.close_np, timeperiod=5)
        result2 = self.df_pl.select(
            pl.col('close').ta.BBANDS(timeperiod=5, schema=['upperband', 'middleband', 'lowerband'], skip_nan=False).alias('BBANDS')
        ).unnest('BBANDS')
        result3 = result2['upperband'].to_numpy()
        # print()
        # print(result1)
        # print(result3)

        assert np.allclose(result1, result3, equal_nan=True)
