"""
复权算法
1. 针对现金分红进行复权（使用加减法）
2. 针对拆股进行复权（使用乘除法）
"""


def calc_factor_muldiv(df: pl.DataFrame,
                       by1: str = 'stock_code', by2: str = 'time',
                       close: str = 'close', pre_close: str = 'pre_close') -> pl.DataFrame:
    """计算复权因子，乘除法。使用交易所发布的昨收盘价计算

    Parameters
    ----------
    df : pl.DataFrame
        数据
    by1 : str
        分组字段
    by2 : str
        排序字段
    close : str
        收盘价字段
    pre_close : str
        昨收盘价字段

    Notes
    -----
    不关心是否真发生了除权除息过程，只要知道前收盘价和收盘价不等就表示发生了除权除息

    """
    df = (
        df
        .sort(by1, by2)
        .with_columns(factor1=(pl.col(close).shift(1, fill_value=pl.first(pre_close)) / pl.col(pre_close)).round(8).over(by1, order_by=by2))
        .with_columns(factor2=(pl.col('factor1').cum_prod()).over(by1, order_by=by2))
    )
    return df


def calc_factor_addsub(df: pl.DataFrame,
                       by1: str = 'stock_code', by2: str = 'time',
                       close: str = 'close', pre_close: str = 'pre_close') -> pl.DataFrame:
    """计算复权因子，加减法。使用交易所发布的昨收盘价计算

    Parameters
    ----------
    df : pl.DataFrame
        数据
    by1 : str
        分组字段
    by2 : str
        排序字段
    close : str
        收盘价字段
    pre_close : str
        昨收盘价字段

    Notes
    -----
    不关心是否真发生了除权除息过程，只要知道前收盘价和收盘价不等就表示发生了除权除息

    """
    df = (
        df
        .sort(by1, by2)
        .with_columns(factor1=(pl.col(close).shift(1, fill_value=pl.first(pre_close)) - pl.col(pre_close)).round(8).over(by1, order_by=by2))
        .with_columns(factor2=(pl.col('factor1').cum_sum()).over(by1, order_by=by2))
    )
    return df
