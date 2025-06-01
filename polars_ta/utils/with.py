import re

import polars as pl


def with_industry(df: pl.DataFrame, industry_name: str, *, drop_first: bool) -> pl.DataFrame:
    """添加行业哑元变量

    Parameters
    ----------
    df
    industry_name:str
        行业列名
    drop_first
        丢弃第一列

    Returns
    -------
    pl.DataFrame

    """
    df = df.with_columns([
        # 行业处理，由浮点改成整数
        pl.col(industry_name).cast(pl.UInt32),
    ])

    # TODO 没有行业的也过滤，这会不会有问题？已经退市了，但引入了未来数据
    df = df.filter(
        pl.col(industry_name).is_not_null(),
    )
    # drop_first丢弃哪个字段是随机的，非常不友好，只能在行业中性化时动态修改代码
    df = df.to_dummies(industry_name, drop_first=False)
    if drop_first:
        industry_columns = sorted(filter(lambda x: re.search(rf"^{industry_name}_\d+$", x), df.columns))
        return df.drop(industry_columns[0])
    else:
        return df
