import numpy as np
import polars as pl
from pandas.testing import assert_frame_equal, assert_series_equal


class TestDemoClass:
    high_np = None
    low_np = None
    close_np = None
    df_pl = None

    def setup_class(self):
        self.high_np = np.random.rand(100)
        self.low_np = np.arange(100, dtype=float) - np.random.rand(100)
        self.close_np = np.arange(100, dtype=float)

        self.df_pl = pl.DataFrame([self.high_np, self.low_np, self.close_np],
                                  schema=["high", "low", "close"])
        self.df_pl = self.df_pl.with_columns(pl.lit(None).alias("null"))
        self.df_pd = self.df_pl.to_pandas()

    def test_ts_rank(self):
        from polars_ta.wq.time_series import ts_rank

        result1 = self.df_pd[["high", 'low']].rolling(5).rank(pct=True)
        result2 = self.df_pl.select(ts_rank(pl.col(["high", 'low']), d=5))

        # 底层一样，结果就应当一样
        assert_frame_equal(result1, result2.to_pandas())

    def test_ts_skewness(self):
        from polars_ta.wq.time_series import ts_skewness

        # 好像没办法告诉pandas关闭偏差校正
        result1 = self.df_pd[["high"]].rolling(5).skew()
        result2 = self.df_pl.select(ts_skewness(pl.col(["high"]), d=5))
        # print(result1)
        # print(result2)

        assert_frame_equal(result1, result2.to_pandas())

    def test_ts_kurtosis(self):
        from polars_ta.wq.time_series import ts_kurtosis

        result1 = self.df_pd[["high"]].rolling(5).kurt()
        result2 = self.df_pl.select(ts_kurtosis(pl.col(["high"]), d=5))

        # 底层一样，结果就应当一样
        assert_frame_equal(result1, result2.to_pandas())

    def test_ts_corr(self):
        from polars_ta.wq.time_series import ts_corr

        result1 = self.df_pd["high"].rolling(5).corr(self.df_pd["low"])
        result2 = self.df_pl.select(ts_corr(pl.col("high"), pl.col("low"), d=5))

        assert_series_equal(result1, result2.to_series(0).to_pandas(), check_names=False)

    def test_ts_covariance(self):
        from polars_ta.wq.time_series import ts_covariance

        result1 = self.df_pd["high"].rolling(5).cov(self.df_pd["low"])
        result2 = self.df_pl.select(ts_covariance(pl.col("high"), pl.col("low"), d=5))

        assert_series_equal(result1, result2.to_series(0).to_pandas(), check_names=False)

    # def test_ts_co_kurtosis(self):
    #     from polars_ta.wq.time_series import ts_covariance
    #
    #     result1 = self.df_pd["high"].rolling(10).kurt(self.df_pd["low"])
    #     print(result1)
    #     # result2 = self.df_pl.select(ts_covariance(pl.col("high"), pl.col("low"), d=5))
    #     #
    #     # assert_series_equal(result1, result2.to_series(0).to_pandas(), check_names=False)

    def test_ts_decay_exp_window(self):
        from polars_ta.wq.time_series import ts_decay_exp_window

        df = pl.DataFrame({'A': range(100)})
        print(df)
        result1 = df.select(ts_decay_exp_window(pl.col('A'), 10, factor=0.5))
        print(result1)

    def test_ts_weighted_delay(self):
        from polars_ta.wq.time_series import ts_weighted_delay

        df = pl.DataFrame({'A': [4, 6]})
        print(df)
        result1 = df.select(ts_weighted_delay(pl.col('A'), k=0.25))
        print(result1)

    # def test_ts_covariance(self):
    #     df_pl = pl.DataFrame({'A': [4, 6, 5, 10], 'B': [7, 5, 3, 1]})
    #     df_pd: pd.DataFrame = df_pl.to_pandas()
    #
    #     # print(df_pl.corr())
    #     # print(df_pd.cov())
    #
    #     print(self.df_pl.select(pl.rolling_cov('high', 'close', window_size=10, ddof=1)))
    #     print(self.df_pl.select(pl.rolling_cov('close', 'high', window_size=10, ddof=1)))
    #     # print(df_pd['A'].rolling(4).cov(df_pd['B']))
    #     print(df_pd['B'].rolling(4).skew(df_pd['A']))
    #     print(df_pd['B'].rolling(4).kurt(df_pd['A']))
    #     df_pl['A'].skew()
    #     df_pl['A'].kurtosis()
