"""
In this file we use
open_: Expr, high: Expr, low: Expr, close: Expr
for all parameters
to ensure the similar function signatures when calling them

本文件中参数全用
open_: Expr, high: Expr, low: Expr, close: Expr
统一的好处是在使用时不用考虑函数调用区别
"""
from polars import Expr
from polars import max_horizontal, min_horizontal

from polars_ta import TA_EPSILON


# https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_utility.h#L327
def real_body(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """实体"""
    return (close - open_).abs()


def upper_shadow(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """上影"""
    return high - max_horizontal(open_, close)


def lower_shadow(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """下影"""
    return min_horizontal(open_, close) - low


def high_low_range(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """总长"""
    return high - low


def upper_body(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """实体上沿"""
    return max_horizontal(open_, close)


def lower_body(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """实体下沿"""
    return min_horizontal(open_, close)


def shadows(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """阴影"""
    return high_low_range(open_, high, low, close) - real_body(open_, high, low, close)


def efficiency_ratio(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """
    abs(close-open) / (2 * (high-low) - abs(close-open) + TA_EPSILON)
    K线内的市场效率。两个总长减去一个实体长就是路程

    比较粗略的计算市场效率的方法。丢失了部分路程信息，所以结果会偏大
    """
    displacement = real_body(open_, high, low, close)
    distance = 2 * high_low_range(open_, high, low, close) - displacement
    return displacement / (distance + TA_EPSILON)


# ====================================

def candle_color(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """K线颜色"""
    return close >= open_


def four_price_doji(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """一字"""
    return low >= (high - TA_EPSILON)


def doji(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """十字(含一字、T字)"""
    return (open_ - close).abs() <= TA_EPSILON


def dragonfly(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """正T字"""
    return doji(open_, high, low, close) & (low < close - TA_EPSILON)


def gravestone(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
    """倒T字"""
    return doji(open_, high, low, close) & (high > close + TA_EPSILON)

# def candle_range(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
#     return real_body(open_, high, low, close)
#     return high_low_range(open_, high, low, close)
#     return shadows(open_, high, low, close)
#
#
# def candle_average(open_: Expr, high: Expr, low: Expr, close: Expr) -> Expr:
#     """平均值"""
#     return high_low_range(open_, high, low, close)
