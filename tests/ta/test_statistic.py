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

    def test_LINEARREG(self):
        from polars_ta.ta.statistic import LINEARREG

        walk_np = 100 + np.cumsum(np.random.default_rng(42).normal(size=300))
        df = pl.DataFrame({"walk": walk_np})

        for timeperiod in (2, 5, 14, 30):
            result1 = talib.LINEARREG(walk_np, timeperiod=timeperiod)
            result2 = df.select(LINEARREG(pl.col("walk"), timeperiod=timeperiod))
            result3 = result2['walk'].to_numpy()

            assert np.allclose(result1, result3, equal_nan=True)
            assert np.allclose(linearreg_lstsq(walk_np, timeperiod), result3, equal_nan=True)

        result1 = talib.LINEARREG(self.high_np)
        result2 = self.df_pl.select(LINEARREG(pl.col("high")))
        result3 = result2['high'].to_numpy()

        assert np.allclose(result1, result3, equal_nan=True)

    def test_LINEARREG_line_and_constant(self):
        from polars_ta.ta.statistic import LINEARREG

        # a straight line is its own regression line
        result = self.df_pl.select(LINEARREG(pl.col("close"), timeperiod=14))['close']
        assert result[:13].is_null().all()
        assert np.allclose(result[13:].to_numpy(), self.close_np[13:])

        df = pl.DataFrame({"a": np.full(20, 7.5)})
        result = df.select(LINEARREG(pl.col("a"), timeperiod=5))['a']
        assert result[:4].is_null().all()
        assert np.allclose(result[4:].to_numpy(), 7.5)

    def test_LINEARREG_warmup(self):
        import polars_ta
        from polars_ta.ta.statistic import LINEARREG

        df = pl.DataFrame({"a": [1.0, 2.0, 4.0, 3.0]})

        # shorter than timeperiod: no complete window
        assert df.select(LINEARREG(pl.col("a"), timeperiod=5))['a'].is_null().all()

        result = df.select(LINEARREG(pl.col("a"), timeperiod=4))['a']
        assert result[:3].is_null().all()
        assert np.isclose(result[3], 3.7)

        # a partial window is not a regression, so MIN_SAMPLES must not shorten the warmup
        old_min_samples = polars_ta.MIN_SAMPLES
        polars_ta.MIN_SAMPLES = 1
        try:
            result = df.select(LINEARREG(pl.col("a"), timeperiod=4))['a']
        finally:
            polars_ta.MIN_SAMPLES = old_min_samples
        assert result[:3].is_null().all()
        assert np.isclose(result[3], 3.7)

    def test_LINEARREG_null(self):
        from polars_ta.ta.statistic import LINEARREG

        values = [1.0, 3.0, 2.0, 5.0, None, 4.0, 6.0, 8.0, 7.0, 9.0]
        df = pl.DataFrame({"a": values})
        result = df.select(LINEARREG(pl.col("a"), timeperiod=3))['a']

        # only windows that contain the missing value are null
        assert result.is_null().to_list() == [True, True, False, False, True, True, True, False, False, False]
        expected = linearreg_lstsq(np.array(values, dtype=float), 3)
        assert np.allclose(result.to_numpy(), expected, equal_nan=True)


def linearreg_lstsq(x, timeperiod):
    """End point of the least-squares line fitted to each complete window"""
    out = np.full(len(x), np.nan)
    a = np.vstack([np.arange(timeperiod), np.ones(timeperiod)]).T
    for i in range(timeperiod - 1, len(x)):
        y = x[i - timeperiod + 1:i + 1]
        if np.isnan(y).any():
            continue
        slope, intercept = np.linalg.lstsq(a, y, rcond=None)[0]
        out[i] = slope * (timeperiod - 1) + intercept
    return out
