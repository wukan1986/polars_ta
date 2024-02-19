from pathlib import Path

import pandas as pd
import polars as pl

from polars_ta.utils.pit import ts_pit
from polars_ta.wq.time_series import ts_mean

PATH_STEP0_INPUT1 = r'M:\data\jqresearch\get_STK_BALANCE_SHEET'


# PATH_STEP0_INPUT1 = r'M:\data\jqresearch\get_fundamentals_cash_flow'
# PATH_STEP0_INPUT1 = r'M:\data\jqresearch\get_fundamentals_income'
# PATH_STEP0_INPUT1 = r'M:\data\jqresearch\get_fundamentals_indicator'


def load_parquet(folder):
    paths = list(Path(folder).glob('*.parquet'))
    # 写入时由于部分数据为空，导致入写类型不同，读取就存在问题，只能用pandas读回来
    dfs = pd.concat([pd.read_parquet(p) for p in paths])
    return pl.from_pandas(dfs, nan_to_null=True)


df1 = load_parquet(PATH_STEP0_INPUT1)


def func(df: pl.DataFrame):
    df = df.with_columns(
        ts_mean(pl.col('id'), 2).alias('test1')
    )
    return df


if __name__ == '__main__':
    d = df1.group_by('code').map_groups(lambda x: ts_pit(x, funcs=(func,), date='report_date', update_time='pub_date'))
    print(d.tail())
