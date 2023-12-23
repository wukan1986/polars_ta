import polars as pl

from polars_ta import TA_EPSILON


# https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_utility.h#L327

def real_body(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """实体"""
    return (close - open_).abs()


def upper_shadow(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """上影"""
    return high - pl.max_horizontal(open_, close)


def lower_shadow(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """下影"""
    return pl.min_horizontal(open_, close) - low


def high_low_range(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """总长"""
    return high - low


def candle_color(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """K线颜色"""
    return close >= open_
    # return pl.when(close >= open_).then(1).otherwise(-1)


def upper_body(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """实体上沿"""
    return pl.max_horizontal(open_, close)


def lower_body(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """实体下沿"""
    return pl.min_horizontal(open_, close)


def shadows(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """阴影"""
    return high_low_range(open_, high, low, close) - real_body(open_, high, low, close)


def efficiency_ratio(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """K线内的市场效率。两个总长减去一个实体长就是路程

    比较粗略的计算市场效率的方法。丢失了部分路程信息，所以结果会偏大
    """
    displacement = real_body(open_, high, low, close)
    distance = 2 * high_low_range(open_, high, low, close) - displacement
    return displacement / (distance + TA_EPSILON)

# def candle_range(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
#     return real_body(open_, high, low, close)
#     return high_low_range(open_, high, low, close)
#     return shadows(open_, high, low, close)
#
#
# def candle_average(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
#     """平均值"""
#     return high_low_range(open_, high, low, close)
