import numpy as np
from polars import Expr

from polars_ta.wq.arithmetic import log1p, expm1  # noqa
# 对数收益
from polars_ta.wq.time_series import ts_log_diff as ts_log_return  # noqa
# 简单收益
from polars_ta.wq.time_series import ts_returns as ts_percent_return  # noqa


def ts_cum_return(close: Expr) -> Expr:
    """累计收益"""

    return close / close.drop_nulls().first()


def simple_to_log_return(x: Expr) -> Expr:
    """简单收益率 转 对数收益率"""
    return log1p(x)


def log_to_simple_return(x: Expr) -> Expr:
    """对数收益率 转 简单收益率"""
    return expm1(x)


def cumulative_returns(returns: np.ndarray, weights: np.ndarray, period: int = 3, is_mean: bool = True) -> np.ndarray:
    """累积收益

    精确计算收益是非常麻烦的事情，比如考虑手续费、滑点、涨跌停无法入场。考虑过多也会导致计算量巨大。
    这里只做估算，用于不同因子之间收益比较基本够用。更精确的计算请使用专用的回测引擎

    需求：因子每天更新，但策略是持仓3天
    1. 每3天取一次因子，并持有3天。即入场时间对净值影响很大。净值波动剧烈
    2. 资金分成3份，每天入场一份。每份隔3天做一次调仓，多份资金不共享。净值波动平滑

    本函数使用的第2种方法，例如：某支股票持仓信息如下
    [0,1,1,1,0,0]
    资金分成三份，每次持有三天，
    [0,0,0,1,1,1] # 第0、3、6...位，fill后两格
    [0,1,1,1,0,0] # 第1、4、7...位，fill后两格
    [0,0,1,1,1,0] # 第2、5、8...位，fill后两格

    之后就是weights*returns就是每期的收益率，横截面mean后就是这份资金每天的收益率。+1再cumprod就是这份资金的净值
    最后多份资金直接平均，就是总的净值

    weights*returns做了period轮
    cumprod计算了period次

    Parameters
    ----------
    returns: np.ndarray
        1期简单收益率。自动记在出场位置。
    weights: np.ndarray
        持仓权重。需要将信号移动到出场日期
    period: int
        持有期数。即资金拆成多少份
    is_mean:
        权重处理方式。
        - True 表示等权，weights的取值为-1,0,1
        - False 表示在外指定权重。weights的取值为-1~1,weights.abs().sum(axis=1)==1

    Returns
    -------
    np.ndarray

    References
    ----------
    https://github.com/quantopian/alphalens/issues/187

    """
    # 修正数据中出现的nan
    returns = np.where(returns == returns, returns, 0.0)
    weights = np.where(weights == weights, weights, 0.0)
    # 一维修改成二维，代码统一
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    #
    m, n = weights.shape
    #  记录每份资金的净值
    out = np.zeros(shape=(m, period), dtype=float)
    for i in range(period):
        # 初始化新持仓
        w = np.zeros_like(weights)
        # 某一天的持仓需要持续period天
        a = np.repeat(weights[i::period], period, axis=0)
        # 更新到指定位置
        w[i:] = a[:m - i]
        # 计算此份资金的收益，净值从1开始
        if is_mean:
            # 等权分配
            out[:, i] = (returns * w).mean(axis=1)
        else:
            # 将权重分配交给外部。一般pos.abs.sum==1
            out[:, i] = (returns * w).sum(axis=1)

    # 多份净值直接叠加平均
    return (out + 1).cumprod(axis=0).mean(axis=1)
