from pathlib import Path

import pandas as pd
import polars as pl

from polars_ta.utils.pit import ts_pit, period_to_quarter, peroid_to_ttm, point_to_ttm

PATH_STEP0_INPUT1 = r'M:\data\jqresearch\get_STK_BALANCE_SHEET'
PATH_STEP0_INPUT2 = r'M:\data\jqresearch\get_STK_INCOME_STATEMENT'
PATH_STEP0_INPUT3 = r'M:\data\jqresearch\get_STK_CASHFLOW_STATEMENT'


def load_parquet(folder):
    paths = list(Path(folder).glob('*.parquet'))
    # 写入时由于部分数据为空，导致入写类型不同，读取就存在问题，只能用pandas读回来
    dfs = pd.concat([pd.read_parquet(p) for p in paths])
    return pl.from_pandas(dfs, nan_to_null=True)


df1 = load_parquet(PATH_STEP0_INPUT1)
df1 = df1.filter(pl.col('report_type') == 0)
df2 = load_parquet(PATH_STEP0_INPUT2)
df2 = df2.filter(pl.col('report_type') == 0)
df3 = load_parquet(PATH_STEP0_INPUT3)
df3 = df3.filter(pl.col('report_type') == 0)


def func1(df: pl.DataFrame):
    """资产负债表"""
    df = df.with_columns(
        point_to_ttm()
    )
    return df


def func2(df: pl.DataFrame, date='report_date', update_time='pub_date', asset='code'):
    """
    利润表，现金流量表
    """
    df1 = df.with_columns(
        # 转成单季
        period_to_quarter(),
        peroid_to_ttm(),
    )
    return df1


if __name__ == '__main__':
    d1 = df1.group_by('code').map_groups(lambda x: ts_pit(x, func=func1,
                                                          date='report_date', update_time='pub_date', asset='code'))
    d2 = df2.group_by('code').map_groups(lambda x: ts_pit(x, func=func2,
                                                          date='report_date', update_time='pub_date', asset='code'))
    d3 = df3.group_by('code').map_groups(lambda x: ts_pit(x, func=func2,
                                                          date='report_date', update_time='pub_date', asset='code'))
    print(d3.tail().to_pandas())
