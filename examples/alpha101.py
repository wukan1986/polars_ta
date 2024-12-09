from datetime import datetime

import pandas as pd
import polars as pl

from polars_ta.prefix.tdx import *
from polars_ta.prefix.wq import *

# definition of your features
# 因子定义
OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT, = pl.col('OPEN'), pl.col('HIGH'), pl.col('LOW'), pl.col('CLOSE'), pl.col('VOLUME'), pl.col('AMOUNT'),
RETURNS, VWAP, CAP, = pl.col('RETURNS'), pl.col('VWAP'), pl.col('CAP'),
ADV5, ADV10, ADV15, ADV20, ADV30, ADV40, ADV50, ADV60, ADV81, ADV120, ADV150, ADV180, = (
    pl.col('ADV5'), pl.col('ADV10'), pl.col('ADV15'), pl.col('ADV20'),
    pl.col('ADV30'), pl.col('ADV40'), pl.col('ADV50'), pl.col('ADV60'),
    pl.col('ADV81'), pl.col('ADV120'), pl.col('ADV150'), pl.col('ADV180'),)
SECTOR, INDUSTRY, SUBINDUSTRY, = pl.col('SECTOR'), pl.col('INDUSTRY'), pl.col('SUBINDUSTRY'),

# you can only call column-alike factor directly. Since ts_, cs_ and gp_ etc. need to be used with group_by, please use expr_codegen project for more complex formula
# 列因子才可以直接调用，而ts_、cs_和gp_等公式需要套用group_by，复杂公式请使用expr_codegen项目
alpha_041 = (((HIGH * LOW) ** 0.5) - VWAP)
alpha_054 = ((-1 * ((LOW - CLOSE) * (OPEN ** 5))) / ((LOW - HIGH) * (CLOSE ** 5)))
alpha_101 = ((CLOSE - OPEN) / ((HIGH - LOW) + 0.001))

# simulate 5000 stocks for 10 years
# 模拟5000支股票10年
ASSET_COUNT = 5000
DATE_COUNT = 250 * 10
DATE = pd.date_range(datetime(2020, 1, 1), periods=DATE_COUNT).repeat(ASSET_COUNT)
ASSET = [f'A{i:04d}' for i in range(ASSET_COUNT)] * DATE_COUNT

# your test data. multi-asset multi-feature
# 测试数据。多资产多特征
df = pl.DataFrame(
    {
        'date': DATE,
        'asset': ASSET,
        "OPEN": np.random.rand(DATE_COUNT * ASSET_COUNT),
        "HIGH": np.random.rand(DATE_COUNT * ASSET_COUNT),
        "LOW": np.random.rand(DATE_COUNT * ASSET_COUNT),
        "CLOSE": np.random.rand(DATE_COUNT * ASSET_COUNT),
        "VWAP": np.random.rand(DATE_COUNT * ASSET_COUNT),
        "FILTER": np.tri(DATE_COUNT, ASSET_COUNT, k=-2).reshape(-1),
    }
)

# filter out some data. test if it will raise error when the length is not enough
# 每行数据量不同，测试是否会因为长度不足报错
df = df.filter(pl.col("FILTER") == 1)

# some simple features defined in Alpha101. They don't contain groupby in time-series or cross-section
# 部分Alpha101计算。不涉及时序和横截面，可直接计算
df = df.with_columns(
    alpha_041=alpha_041,
    alpha_054=alpha_054,
    alpha_101=alpha_101,
)


def func_ts_date(df: pl.DataFrame) -> pl.DataFrame:
    # ensure sorted before computing time-series features
    # 时序指标计算前一定要保证有序
    df = df.sort(by=['date'])
    # Example of generating features
    # 演示生成大量指标
    df = df.with_columns([
        # load from sub-package `wq`
        # 从wq中导入指标
        *[ts_returns(CLOSE, i).alias(f'ROCP_{i:03d}') for i in (1, 3, 5, 10, 20, 60, 120)],
        *[ts_mean(CLOSE, i).alias(f'SMA_{i:03d}') for i in (5, 10, 20, 60, 120)],
        *[ts_std_dev(CLOSE, i).alias(f'STD_{i:03d}') for i in (5, 10, 20, 60, 120)],
        *[ts_max(HIGH, i).alias(f'HHV_{i:03d}') for i in (5, 10, 20, 60, 120)],
        *[ts_min(LOW, i).alias(f'LLV_{i:03d}') for i in (5, 10, 20, 60, 120)],
        *[ts_rank(CLOSE, i).alias(f'RANK_{i:03d}') for i in (5, 10, 20, 60, 120)],
        *[ts_arg_max(HIGH, i).alias(f'HHVBAR_{i:03d}') for i in (5, 10, 20, 60, 120)],

        # load from sub-package `tdx`
        # 从tdx中导入指标
        *[ts_RSI(CLOSE, i).alias(f'RSI_{i:03d}') for i in (6, 12, 24)],
    ])

    return df


# ensure grouped before applying to multi-asset data
# 多资产需要先按资产分组
df = df.group_by('asset').map_groups(func_ts_date)

print(df)
