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
import polars as pl

# 公布日。收盘后公布将会标记为下一日
# 其实行情数据的日期就是报告日，隐去了公布日
ANNOUNCE_DATE = '__ANNOUNCE_DATE__'


def pit_prepare(df: pl.DataFrame,
                by1: str = 'asset',
                by2: str = 'announce_date',
                by3: str = 'report_date') -> pl.DataFrame:
    """将原始的财务表根据公布日重新扩展，输出满足以下条件：

    1. 根据股票和公布日分组
    2. 组内没有未来数据
    3. 组内的同一报告期数据只取最新

    Returns
    -------
    pl.DataFrame
        - asset
        - announce_date
        - report_date
        - __ANNOUNCE_DATE__
            是公布日也是观察日，在观察日可以看到最新的历史数据，看不到未来数据

    """
    TMP = '__TMP__'

    # 根据发布日期，笛卡尔扩展，过滤掉未来数据
    df1 = (
        df
        .select(pl.col(by1).alias(TMP), pl.col(by2).alias(ANNOUNCE_DATE))
        .join_where(df, (pl.col(TMP) == pl.col(by1)) & (pl.col(ANNOUNCE_DATE) >= pl.col(by2)))
        .drop(TMP)
    )

    # 过滤数据，同报告期只保留最新的一条
    df2 = (
        df1
        .sort(by1, ANNOUNCE_DATE, by3, by2)
        .unique([by1, ANNOUNCE_DATE, by3], keep="last", maintain_order=True)
    )
    return df2


def pit_calc(df: pl.DataFrame,
             by1: str = 'asset',
             by2: str = ANNOUNCE_DATE,
             by3: str = 'report_date') -> pl.DataFrame:
    """输入PIT分组的财务数据，组内计算时序指标"""
    df1 = (
        df.with_columns(
            # TODO 补充其他各种时序指标，注意，不要少了`( ).over(by1, by2, order_by=by3)`
            net_profit_to_total_operate_revenue_ttm=(pl.col('net_profit').rolling_mean(4) / pl.col('total_operating_revenue').rolling_mean(4)).over(by1, by2, order_by=by3)
        )
    )
    return df1


def pit_frist(df: pl.DataFrame,
              by1: str = 'asset',
              by2: str = 'announce_date',
              by3: str = 'report_date',
              by4: str = ANNOUNCE_DATE) -> pl.DataFrame:
    """输入PIT分组的财务数据，多组合并保留最先发布的数据

    此数据不含未来数据，但原则上不能再做时序处理

    Returns
    -------
    pl.DataFrame
        - asset
        - announce_date
        - report_date

    """
    df1 = (
        df
        .sort(by1, by2, by3, by4)
        .unique([by1, by2, by3], keep="first", maintain_order=True)
        .drop(ANNOUNCE_DATE)
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


def join_quote_financial(quote: pl.DataFrame,
                         financial: pl.DataFrame,
                         by1: str = 'asset',
                         by2: str = 'date'):
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
