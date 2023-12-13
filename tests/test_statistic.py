import numpy as np
import polars as pl
import talib


class TestDemoClass:
    close_np = None
    close_pl = None

    def setup_class(self):
        self.close_np = np.arange(10, dtype=float)
        self.close_pl = pl.DataFrame(self.close_np, schema=["close"])

    def test_STDDEV(self):
        from polars_ta.talib.statistic import STDDEV

        result1 = talib.STDDEV(self.close_np, timeperiod=3)
        result2 = self.close_pl.select(STDDEV(pl.col("close"), timeperiod=3))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)

    def test_VAR(self):
        from polars_ta.talib.statistic import VAR

        result1 = talib.VAR(self.close_np, timeperiod=5)
        result2 = self.close_pl.select(VAR(pl.col("close"), timeperiod=5))
        result3 = result2['close'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)
