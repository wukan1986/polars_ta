from pathlib import Path

import pandas as pd
import polars as pl

from polars_ta.utils.pit import pit_prepare, ttm_from_point, ANNOUNCE_DATE, pit_frist, ttm_from_period

PATH_STEP0_INPUT1 = r'F:\data\jqresearch\get_STK_BALANCE_SHEET'
PATH_STEP0_INPUT2 = r'F:\data\jqresearch\get_STK_INCOME_STATEMENT'
PATH_STEP0_INPUT3 = r'F:\data\jqresearch\get_STK_CASHFLOW_STATEMENT'


def load_parquet(folder):
    paths = list(Path(folder).glob('*.parquet'))
    # due to the data is not complete when writing, the data type can only be read from pandas then convert to polars
    # 写入时由于部分数据为空，导致入写类型不同，读取就存在问题，只能用pandas读回来
    dfs = pd.concat([pd.read_parquet(p) for p in paths])
    return pl.from_pandas(dfs, nan_to_null=True)


def balance():
    # https://www.joinquant.com/help/api/help#Stock:合并资产负债表
    df1 = load_parquet(PATH_STEP0_INPUT1)
    df1 = df1.filter(pl.col('report_type') == 0)
    df1 = pit_prepare(df1, by1='code', by2='pub_date', by3='report_date')
    df2 = df1.select(
        "code", "pub_date", "report_date", ANNOUNCE_DATE,
        "total_assets", ttm_from_point('total_assets').over('code', ANNOUNCE_DATE, order_by='report_date').name.suffix('_ttm'),
        "equities_parent_company_owners", ttm_from_point('equities_parent_company_owners').over('code', ANNOUNCE_DATE, order_by='report_date').name.suffix('_ttm'),
    )
    df3 = pit_frist(df2, by1='code', by2='pub_date', by3='report_date', by4=ANNOUNCE_DATE)
    return df3


def income():
    # https://www.joinquant.com/help/api/help#Stock:合并利润表
    df1 = load_parquet(PATH_STEP0_INPUT2)
    df1 = df1.filter(pl.col('report_type') == 0)
    df1 = df1.with_columns(quarter=pl.col('report_date').dt.quarter())
    df1 = pit_prepare(df1, by1='code', by2='pub_date', by3='report_date')
    df2 = df1.select(
        "code", "pub_date", "report_date", ANNOUNCE_DATE,
        "total_operating_revenue", ttm_from_period("total_operating_revenue", quarter='quarter').over('code', ANNOUNCE_DATE, order_by='report_date').name.suffix('_ttm'),
        "net_profit", ttm_from_period('net_profit', quarter='quarter').over('code', ANNOUNCE_DATE, order_by='report_date').name.suffix('_ttm'),
    )
    df3 = pit_frist(df2, by1='code', by2='pub_date', by3='report_date', by4=ANNOUNCE_DATE)
    return df3


def cashflow():
    # https://www.joinquant.com/help/api/help#Stock:合并现金流量表
    df1 = load_parquet(PATH_STEP0_INPUT3)
    df1 = df1.filter(pl.col('report_type') == 0)
    df1 = df1.with_columns(quarter=pl.col('report_date').dt.quarter())
    df1 = pit_prepare(df1, by1='code', by2='pub_date', by3='report_date')
    df2 = df1.select(
        "code", "pub_date", "report_date", ANNOUNCE_DATE,
        "net_operate_cash_flow", ttm_from_period("net_operate_cash_flow", quarter='quarter').over('code', ANNOUNCE_DATE, order_by='report_date').name.suffix('_ttm'),
        "financial_cost", ttm_from_period('financial_cost', quarter='quarter').over('code', ANNOUNCE_DATE, order_by='report_date').name.suffix('_ttm'),
    )
    df3 = pit_frist(df2, by1='code', by2='pub_date', by3='report_date', by4=ANNOUNCE_DATE)
    return df3


if __name__ == '__main__':
    balance()
    income()
    cashflow()
