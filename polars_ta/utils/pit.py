import polars as pl


def ts_pit(df: pl.DataFrame, funcs=(), date='date', update_time='update_time'):
    """Point In Time计算

    Parameters
    ----------
    df: pl.DataFrame
        group_by(date)后的数据块
    funcs:
        作用于pit提取出的DataFrame，可不填
    date:str
        分组依据
    update_time:str
        分组依据

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    1. 数据更新可能是周末
    2. 输出的数据不能再做时序计算，因为再算就引入未来数据了
    3. 此算子与一般的`ts_`算子不同，小心使用

    """
    # 一定要提前排序
    sort_by = [date, update_time]
    df = df.sort(sort_by)

    df2 = (
        # 分组，记录数量，和更新时间
        df.group_by(date).agg(update_time, pl.count())
        # 多加一行是否最后值
        .filter(pl.col('count') > 1)
        .select(pl.col(update_time).list.slice(1))
        .explode(update_time).to_series()
    )

    # 最大的更新时间
    max_update_time = df.select(update_time).max().to_series()
    df3 = df2.append(max_update_time).unique().sort().to_list()

    if len(df3) <= 1:
        # 只有一条，表示中间没有修改过，可直接计算后返回
        for func in funcs:
            df = func(df)
        return df

    dd = []
    # 遍历关键日期
    for dt in df3:
        # 每次只看指定更新日之间的记录
        d = df.filter(pl.col(update_time) <= dt)
        # 已经排序了，只取能取最新值
        d = d.group_by(date, maintain_order=True).last()
        # 分块计算
        for func in funcs:
            d = func(d)
        dd.append(d)

    # 按报告期排序
    x = pl.concat(dd)
    x = x.unique(subset=sort_by, keep='first').sort(sort_by)

    return x
