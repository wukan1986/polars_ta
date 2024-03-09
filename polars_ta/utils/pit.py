import polars as pl
from polars import selectors as cs


def ts_pit(df: pl.DataFrame, func=None,
           date='date', update_time='update_time', asset='asset'):
    """Computing Point In Time

    Parameters
    ----------
    df: pl.DataFrame
        dataframe after group_by(date)
    func:
        apply to the dataframe output from pit, can be None
    date:str
        column name, group by this col
    update_time:str
        column name, group by this col
    asset:str
        asset column name
        资产。asfreq时，asset为空，后面要分组时不方便

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    1. Data update might be weekeng
    2. You cannot use this for further time-series computation.
        Alert for inclusion of future data
    3. This is different from general `ts_` functions, be careful


    Point In Time计算

    Parameters
    ----------
    df: pl.DataFrame
        group_by(date)后的数据块
    func:
        作用于pit提取出的DataFrame，可不填
    date:str
        分组依据
    update_time:str
        分组依据
    asset:str
        资产。asfreq时，asset为空，后面要分组时不方便

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    1. 数据更新可能是周末
    2. 输出的数据不能再做时序计算，因为再算就引入未来数据了
    3. 此算子与一般的`ts_`算子不同，小心使用

    """
    # sort first
    # 一定要提前排序
    sort_by = [date, update_time]
    df = df.sort(sort_by)

    dr = (
        # group by, record the count and update time
        # 分组，记录数量，和更新时间
        df.group_by(date).agg(update_time, pl.len())
        # 多加一行是否最后值
        .filter(pl.col('len') > 1)
        .select(pl.col(update_time).list.slice(1))
        .explode(update_time).to_series()
    )

    # 最大的更新时间
    max_update_time = df.select(update_time).max().to_series()
    dr = dr.append(max_update_time).unique().sort().to_list()

    # fill missing data to 4 quarters
    # 部分很早的财报有缺失，比如只有年报或半年报，仿asfreq补充成4季
    min_max_q = df.select(date).unique().upsample(date, every='1q')
    # TODO 使用date填充update_time是否会引入问题？
    df = df.join(min_max_q, on=date, how='outer_coalesce').with_columns(pl.col(update_time).fill_null(pl.col(date)))
    # 由于asfreq导致asset可能为空，但之后可能要用到，所以填充一下
    df = df.with_columns(pl.col(asset).forward_fill())

    dd = []
    # iterate over the key dates
    # 遍历关键日期
    for dt in dr:
        # 每次只看指定更新日之间的记录
        d: pl.DataFrame = df.filter(pl.col(update_time) <= dt)
        # 已经排序了，只取能取最新值
        d = d.group_by(date, maintain_order=True).last()
        # TODO 其实asfreq放这更合理，但为了提速将其提前

        d = d.with_columns(pl.col(date).dt.quarter().alias('quarter'))
        if func is not None:
            # 分块计算
            d = func(d)
        dd.append(d)

    # sort by the date and update time
    # 按报告期排序
    x = pl.concat(dd)
    x = x.unique(subset=sort_by, keep='first').sort(sort_by)

    return x


def period_to_quarter():
    """时段数据转单成季"""
    csnum = cs.numeric().exclude('quarter')
    return pl.when(pl.col('quarter') == 1).then(csnum).otherwise(csnum.diff()).name.suffix('_Q')


def peroid_to_ttm():
    """时段数据计算ttm"""
    csnum = cs.numeric().exclude('quarter')
    # 年报数据
    f1 = pl.when(pl.col('quarter') == 4).then(csnum).otherwise(None)
    # 上年年报和同比都存在。当前报告期-上年报告期+上年年报
    f2 = csnum - csnum.shift(4) + f1.forward_fill()
    # 年化计算法=当前报告期*年化系数
    f3 = csnum / pl.col('quarter') * 4
    # return pl.coalesce(f1, f2, f3).name.suffix('_TTM')
    return (pl.when(f1.is_not_null()).then(f1)
            .when(f2.is_not_null()).then(f2)
            .when(f3.is_not_null()).then(f3)
            .otherwise(None)).name.suffix('_TTM')


def point_to_ttm() -> pl.Expr:
    """时点数据计算ttm"""
    csnum = cs.numeric().exclude('quarter')
    return csnum.rolling_mean(4).name.suffix('_TTM')
