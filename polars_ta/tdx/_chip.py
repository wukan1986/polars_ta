import numba
import numpy as np


@numba.jit(nopython=True, nogil=True, fastmath=True, cache=True)
def nb_chip(high, low, avg, turnover,
            start=None, stop=None, step=0.2):
    """筹码分布，可用于WINNER或COST指标

    不可能完全还原真实的筹码分布，只能接近。所以做了一下特别处理

    1. 三角分布，比平均分布更接近
    2. 步长。没有必要每个价格都统计，特别是复权后价格也无法正好是0.01间隔
        高价股建议步长设大些，低价股步长需设小些

    Parameters
    ----------
    high
    low
    avg
        一维序列
    turnover:
        换手率，需要在外转成0~1范围内
    start
        开始价格
    stop
        结束价格
    step
        步长。一字涨停时，三角分布的底为1，高为2。但无法当成梯形计算面积，所以从中用半步长切开计算

    Returns
    -------
    out
        筹码分布
    columns
        价格表头

    """
    # 网格范围
    if start is None:
        start = np.min(low)
    if stop is None:
        stop = np.max(high)

    left = round(start / step) * 2 - 1
    right = round(stop / step) * 2 + 1

    # 最小最大值左右要留半格，range是左闭右开，长度必须为2n+1
    columns = np.arange(left, right + 1)
    grid_shape = (len(turnover), len(columns))

    # numba中round写法特殊
    _high = np.empty_like(high)
    _low = np.empty_like(low)
    _avg = np.empty_like(avg)

    # high和low必须落到边缘上
    _high = np.round(high / step, 0, _high) * 2 + 1
    _low = np.round(low / step, 0, _low) * 2 - 1
    # avg必须落在实体上
    _avg = np.round(avg / step, 0, _avg) * 2
    tri_height = 2 / ((_high - _low) // 2)  # 三角形高度

    # 得到三组值在网格中的位置
    high_arg = np.argwhere(columns == _high.reshape(-1, 1))[:, 1]
    avg_arg = np.argwhere(columns == _avg.reshape(-1, 1))[:, 1]
    low_arg = np.argwhere(columns == _low.reshape(-1, 1))[:, 1]

    # 高度表
    height = np.zeros(grid_shape)
    for i in range(len(height)):
        la = low_arg[i]
        aa = avg_arg[i]
        ha = high_arg[i]
        th = tri_height[i]
        height[i, la:aa + 1] = np.linspace(0, th, aa - la + 1)
        height[i, aa:ha + 1] = np.linspace(th, 0, ha - aa + 1)

    # 计算半块面积, 三角形的高变成了梯形的上下底，梯形高固定为0.5，*0.5/2=/4
    # 宽度-1，例如，原长度为5，-1后为4
    area = (height[:, :-1] + height[:, 1:]) / 4
    # 合成一块。宽度/2，例如原长度为4，/2后为2
    weight = area[:, ::2] + area[:, 1::2]

    # 输出
    out = np.zeros_like(weight)
    # 剩余换手率
    turnover2 = 1 - turnover
    # 第一天其实应当用上市发行价，过于麻烦，还是将第一天等权
    # 取巧方法，利用-1的特性，可减少if判断，
    out[-1] = weight[0]
    # 这里现在用的numpy, 还要快可考虑numba
    for i in range(len(turnover)):
        out[i] = out[i - 1] * turnover2[i] + weight[i] * turnover[i]

    # print(out.sum(axis=1))
    return out, (step / 2) * columns[1::2]


@numba.jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _WINNER_COST(high, low, avg, turnover, close, cost, step):
    out, columns = nb_chip(high, low, avg, turnover, step=step)

    # WINNER
    cheap = np.where(columns <= close.reshape(-1, 1), out, 0)
    sum_cheap = np.sum(cheap, axis=1)

    # COST
    # cum = np.cumsum(out, axis=1)
    cum = np.copy(out)
    for i in range(0, out.shape[0]):
        cum[i, :] = np.cumsum(out[i, :])

    prices = np.where(cum <= cost.reshape(-1, 1), columns, 0)

    # np.max(prices, axis=1)
    max_price = prices[:, 0]
    for i in range(0, out.shape[0]):
        max_price[i] = np.max(prices[i, :])

    return sum_cheap, max_price
