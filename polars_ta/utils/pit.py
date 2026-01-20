"""
# Point In Time处理相关函数

## 原始表
资产负债表   时点数据
   利润表   区间数据
现金流量表   区间数据

## 单季度数据
资产负债表   同原始表
   利润表   本季-上季；一季度
现金流量表   本季-上季；一季度

## TTM数据
资产负债表   四个报告期平均值；最新和同比两期的平均值；最新
   利润表   年报；本季+去年报-同比；年化
现金流量表   年报；本季+去年报-同比；年化


## 整理成PIT数据
例如：pit_prepare的功能为
1. ACBD 通过join_where成
2. A,AB,ACB,ACBD 通过sort+unique成
3. A,AB,ABC,ABCD

这样，每一组都满足以观察日视角没有未来数据，并只观察所能看到的最新值，可参考pit_calc示例实现其它处理

处理好的数据再pit_frist取最早公布的项目
例如：A1,A2,A3B3C3,C4D4 取first的结果为 A1B3C3D4

这时的数据就可以做因子了，但不能做时序计算，时序计算得回到pit_calc中实现

"""
from typing import Tuple

import polars as pl

# 公布日。收盘后公布将会标记为下一日
# 其实行情数据的日期相当于报告日report_date，隐去了公布日announce_date，遇到要修正历史数据时才有公布日
LOOKBACK_DATE = '__LOOKBACK_DATE__'


def pit_prepare(df: pl.DataFrame | pl.LazyFrame,
                by1: str = 'asset',
                by2: str = 'report_date',
                by3: str = 'announce_date',
                by4: str = LOOKBACK_DATE,
                lookback_year: str = '-5y') -> pl.DataFrame | pl.LazyFrame:
    """将原始的财务表根据公布日重新扩展，输出满足以下条件：

    1. 根据股票和公布日分组
    2. 组内没有未来数据
    3. 组内的同一报告期数据只取最新

    Returns
    -------
    pl.DataFrame|pl.LazyFrame
        - asset
        - report_date
        - announce_date
        - LOOKBACK_DATE
            是公布日也是观察日，在观察日可以看到最新的历史数据，看不到未来数据

    """

    def upsample_by_cum_max(df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """分割数据，并将单调递增的数据进行重采样"""
        # 取递增
        df1 = df.filter((pl.col(by2) == pl.col(by2).cum_max()).over(by1, order_by=[by3, by2]))

        # 6月30或9月30为一条数据时，upsample会是3月30，12月30，强行改成下月1日
        df1 = df1.with_columns(pl.col(by2).dt.offset_by(by='1d'))
        df1 = df1.sort(by1, by2).upsample(by2, every='1q', group_by=by1).with_columns(pl.col(by1, by3).backward_fill())
        # 还原成上月底
        df1 = df1.with_columns(pl.col(by2).dt.offset_by(by='-1d'))

        # 取非递增
        df2 = df.filter((pl.col(by2) < pl.col(by2).cum_max()).over(by1, order_by=[by3, by2]))
        return df1, df2

    # =========================================
    # ttm等操作shift(4)假定4期报告没有缺失，需要提前重采样补全
    df2 = df.sort(by1, by3, by2)  # 按发布日排序，过滤单调递增
    if isinstance(df, pl.LazyFrame):
        df2 = df2.collect()

    dfs = []
    while True:
        if df2.is_empty():
            break
        df1, df2 = upsample_by_cum_max(df2)
        dfs.append(df1)
    # 合并
    df3 = pl.concat(dfs)
    if isinstance(df, pl.LazyFrame):
        df3 = df3.lazy()

    del df1
    del df2
    del df
    del dfs
    # =========================================
    # 根据发布日期，笛卡尔扩展，过滤掉未来数据
    tmp1 = '__TMP_1__'
    tmp2 = '__TMP_2__'
    df1 = df3.select(pl.col(by1).alias(tmp1), pl.col(by3).dt.offset_by(lookback_year).alias(tmp2), pl.col(by3).alias(by4)).unique()
    df1 = df1.join_where(df3,
                         (pl.col(tmp1) == pl.col(by1))
                         & (pl.col(by4) >= pl.col(by3))
                         & (pl.col(tmp2) <= pl.col(by2))  # 最多观察lookback_year年前报告，减少计算量
                         ).drop(tmp1, tmp2)
    del df3
    # =========================================
    # 过滤数据，同报告期只保留最新的一条
    df1 = (
        df1
        .sort(by1, by4, by2, by3)
        .unique([by1, by4, by2], keep="last", maintain_order=True)
    )

    return df1


def pit_calc(df: pl.DataFrame | pl.LazyFrame,
             by1: str = 'asset',
             by2: str = 'report_date',
             by4: str = LOOKBACK_DATE) -> pl.DataFrame | pl.LazyFrame:
    """输入PIT分组的财务数据，组内计算时序指标

    同观察期下，同一报告期只有一条最新数据，所以没有了by3
    """
    df1 = (
        df.with_columns(
            # TODO 补充其他各种时序指标，注意，不要少了`( ).over(by1, by4, order_by=by2)`
            net_profit_to_total_operate_revenue_ttm=(pl.col('net_profit').rolling_mean(4) / pl.col('total_operating_revenue').rolling_mean(4)).over(by1, by4, order_by=by2)
        )
    )
    return df1


def pit_frist(df: pl.DataFrame | pl.LazyFrame,
              by1: str = 'asset',
              by2: str = 'report_date',
              by3: str = 'announce_date',
              by4: str = LOOKBACK_DATE) -> pl.DataFrame | pl.LazyFrame:
    """输入PIT分组的财务数据，多组合并保留最先发布的数据

    此数据不含未来数据，但原则上不能再做时序处理

    Returns
    -------
    pl.DataFrame|pl.LazyFrame
        - asset
        - report_date
        - announce_date

    Warnings
    --------
    结果不能时序计算，不能取历史，只能取最新

    002509.XSHE 怎么处理？2019年报因故更新晚于2020年一季报，因为一季报先公布，所以2019年报显示永远为null

    """
    df1 = (
        df
        .sort(by1, by4, by2, by3)
        .unique([by1, by2], keep="first", maintain_order=True)
        .drop(by4)  # 处理后by4的值与by2相等，可直接丢弃
    )
    return df1


def period_to_quarter(col: pl.Expr | str, quarter: pl.Expr | str) -> pl.Expr:
    """区间数据转成单季数据

    1. `利润表`和`现金流量表`的`原始表`可以转成单季数据
    2. `资产负债表`不能转单季

    Examples
    --------
    ```python
    df = df.with_columns(
        quarter=pl.col('report_date').dt.quarter(),
    ).with_columns(
        period_to_quarter(cs.numeric().exclude('quarter'), quarter='quarter').over('code', 'pub_date', order_by='report_date').name.suffix('_Q'),
    )
    ```

    """
    col = pl.col(col)
    quarter = pl.col(quarter)
    return pl.when(quarter == 1).then(col).otherwise(col.diff())


def ttm_from_point(col: pl.Expr | str) -> pl.Expr:
    """时点数据计算TTM

    1. 仅`资产负债表`原始表可调用此函数
    2. `利润表`和`现金流量表`不能调用

    数据要按一年四期排列，不能一年两期

    Examples
    --------
    ```python
    df = df.with_columns(
        ttm_from_point('total_assets').over('code', 'pub_date', order_by='report_date').name.suffix('_ttm')
    )
    ```

    """
    col = pl.col(col)
    return pl.coalesce(
        col.rolling_mean(4),  # 4期平均
        (col + col.shift(4)) / 2,  # 同比报告期平均
        col,  # 最新
    )


def last_year(col: pl.Expr | str, quarter: pl.Expr | str) -> pl.Expr:
    """最新年报

    Examples
    --------
    ```python
    df = df.with_columns(
        quarter=pl.col('report_date').dt.quarter(),
    ).with_columns(
        last_year('total_assets').over('code', 'pub_date', order_by='report_date').name.suffix('_ly')
    )
    ```

    """
    col = pl.col(col)
    quarter = pl.col(quarter)
    return pl.when(quarter == 4).then(col).otherwise(None).forward_fill(3)


def ttm_from_period(col: pl.Expr | str, quarter: pl.Expr | str) -> pl.Expr:
    """区间数据计算ttm

    1. `利润表`和`现金流量表`原始表可调用此函数
    2. `资产负债表`不能调用

    数据要按一年四期排列，不能一年两期

    Examples
    --------
    ```python
    df = df.with_columns(
        quarter=pl.col('report_date').dt.quarter(),
    ).with_columns(
        ttm_from_peroid('total_operating_revenue', quarter='quarter').over('code', 'pub_date', order_by='report_date').name.suffix('_TTM')
    )

    ```
    """
    col = pl.col(col)
    quarter = pl.col(quarter)

    # 年报数据
    f1 = pl.when(quarter == 4).then(col).otherwise(None)
    # 当前报告期+上年年报-上年同比
    f2 = col + f1.forward_fill(3) - col.shift(4)
    # 年化计算法=当前报告期*年化系数
    f3 = col / quarter * 4
    return pl.coalesce(f1, f2, f3)


def yoy(col: pl.Expr, period: int = 4) -> pl.Expr:
    """YOY : （Year On Year）同比增长率 """
    return (col - col.shift(period)) / col.shift(period).abs()


def qoq(col: pl.Expr, period: int = 1) -> pl.Expr:
    """QOQ : （Quarter On Quarter） 环比增长率"""
    return (col - col.shift(period)) / col.shift(period).abs()


def join_quote_financial(quote: pl.DataFrame | pl.LazyFrame,
                         financial: pl.DataFrame | pl.LazyFrame,
                         by1: str = 'asset',
                         by2: str = 'date') -> pl.DataFrame | pl.LazyFrame:
    """合并行情与财务表。请提前对财务表进行ttm等计算。因为之后再按报告期对齐很麻烦

    1. 同一天发布多期，需要正确排序最新一期
    2. 更新历史上的某一期，不能把他当成最新一期

    Parameters
    ----------
    quote:
        行情表
    financial:
        财务表。收盘后公布会显示第二天，一周7天都可能公布。同一天能公布多期
    by1
    by2

    """
    quote = quote.sort(by1, by2)
    financial = financial.sort(by1, by2)

    return quote.join_asof(financial, left_on=by2, right_on=by2, by=by1, strategy="backward", check_sortedness=False)
