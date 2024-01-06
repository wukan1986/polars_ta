import time

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
        self.df_pl = self.df_pl.cast(pl.Float32)
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

        assert_frame_equal(result1, result2.to_pandas().astype(float))

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

        # assert_series_equal(result1, result2.to_series(0).to_pandas(), check_names=False)

    def test_ts_covariance(self):
        from polars_ta.wq.time_series import ts_covariance

        result1 = self.df_pd["high"].rolling(5).cov(self.df_pd["low"])
        result2 = self.df_pl.select(ts_covariance(pl.col("high"), pl.col("low"), d=5))

        # assert_series_equal(result1, result2.to_series(0).to_pandas().astype(float), check_names=False)

    def test_ts_decay_exp_window(self):

        df = pl.DataFrame({'A': range(100)})
        print(df)
        # result1 = df.select(ts_decay_exp_window(pl.col('A'), 10, factor=0.5))
        # print(result1)

    def test_ts_decay_linear(self):

        df = pl.DataFrame({'A': [None] * 10, 'B': [None, None, None, None, 4, 5, 6, 7, 8, 9]})
        print(df)
        # result1 = df.select(ts_decay_linear(pl.col('B'), 3))
        # print(result1)

    def test_ts_weighted_delay(self):
        from polars_ta.wq.time_series import ts_weighted_delay

        df = pl.DataFrame({'A': [4, 6]})
        print(df)
        result1 = df.select(ts_weighted_delay(pl.col('A'), k=0.25))
        print(result1)

    def test_ts_arg_max(self):
        from polars_ta.wq._slow import ts_arg_min as func_slow
        from polars_ta.wq.time_series import ts_arg_min as func_fast

        xx = self.df_pl.with_columns(
            a1=func_slow(pl.col('high'), 10),
            a2=func_fast(pl.col('high'), 10),
        )
        print(xx)

        t1 = time.perf_counter()
        for i in range(10000):
            result2 = self.df_pl.with_columns(
                a1=func_slow(pl.col('high'), 10),
            )
        t2 = time.perf_counter()
        print(t2 - t1)

        t1 = time.perf_counter()
        for i in range(10000):
            result2 = self.df_pl.with_columns(
                a1=func_fast(pl.col('high'), 10),
            )
        t2 = time.perf_counter()
        print(t2 - t1)

    def test_ts_product(self):
        from polars_ta.wq._slow import ts_product as func_slow
        from polars_ta.wq.time_series import ts_product as func_fast

        xx = self.df_pl.with_columns(
            a1=func_slow(pl.col('high'), 10),
            a2=func_fast(pl.col('high'), 10),
        )
        print(xx)

        t1 = time.perf_counter()
        for i in range(10000):
            result2 = self.df_pl.with_columns(
                a1=func_slow(pl.col('high'), 10),
            )
        t2 = time.perf_counter()
        print(t2 - t1)

        t1 = time.perf_counter()
        for i in range(10000):
            result2 = self.df_pl.with_columns(
                a1=func_fast(pl.col('high'), 10),
            )
        t2 = time.perf_counter()
        print(t2 - t1)
