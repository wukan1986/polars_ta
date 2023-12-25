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

    def test_EXPMEMA(self):
        from polars_ta.tdx.reference import EXPMEMA

        result1 = talib.EMA(self.close_np, timeperiod=6)
        result2 = self.df_pl.select(EXPMEMA(pl.col("close"), timeperiod=6))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)
