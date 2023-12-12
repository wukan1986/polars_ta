import numpy as np
import polars as pl
import talib


class TestDemoClass:
    high_np = None
    low_np = None
    close_np = None
    df_pl = None

    def setup_class(self):
        self.high_np = np.arange(10, dtype=float) + np.random.rand(10)
        self.low_np = np.arange(10, dtype=float) - np.random.rand(10)
        self.close_np = np.arange(10, dtype=float)

        self.df_pl = pl.DataFrame([self.high_np, self.low_np, self.close_np],
                                  schema=["high", "low", "close"])

    def test_TRANGE(self):
        from polars_ta.volatility import TRANGE

        result1 = talib.TRANGE(self.high_np, self.low_np, self.close_np)
        result2 = self.df_pl.select(TRANGE(pl.col("high"), pl.col("low"), pl.col("close")))
        result3 = result2['max'].to_numpy()
        # print()
        # print(result1)
        # print(result3)

        # pl版第一个位置计算的值为max(x,nan,nan)=x比talib多一个值
        assert np.allclose(result1[1:], result3[1:], equal_nan=True)

    def test_ATR(self):
        # !!!
        talib.set_compatibility(1)

        from polars_ta.volatility import ATR
        result1 = talib.ATR(self.high_np, self.low_np, self.close_np, timeperiod=5)
        result2 = self.df_pl.select(ATR(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=5))
        result3 = result2['max'].to_numpy()
        # print()
        # print(result1)
        # print(result3)

        # # pl版第一个位置计算的值为max(x,nan,nan)=x比talib多一个值
        # assert np.allclose(result1[1:], result3[1:], equal_nan=True)
