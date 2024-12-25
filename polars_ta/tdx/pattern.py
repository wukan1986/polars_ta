from polars import Expr, Struct, Field, Float64, struct

from polars_ta.tdx._chip import _WINNER_COST
from polars_ta.utils.numba_ import batches_i2_o2


def ts_WINNER_COST(high: Expr, low: Expr, avg: Expr, turnover: Expr, close: Expr, cost: Expr = 0.5, step: float = 0.1) -> Expr:
    """
    获利盘比例
        WINNER(CLOSE),表示以当前收市价卖出的获利盘比例,例如返回0.1表示10%获利盘;WINNER(10.5)表示10.5元价格的获利盘比例

    成本分布价
         COST(10),表示10%获利盘的价格是多少,即有10%的持仓量在该价格以下,其余90%在该价格以上,为套牢盘

    Parameters
    ----------
    high
        最高价
    low
        最低价
    avg
        平均价。可以用vwap
    turnover:
        换手率。需要在外转成0~1范围内
    close
        判断获利比例的价格，可以用收盘价，也可以用均价
    cost
        成本比例，0~1
    step
        步长。一字涨停时，三角分布的底为1，高为2。但无法当成梯形计算面积，所以从中用半步长切开计算

    Returns
    -------
    winner
        获利盘比例
    cost
        成本分布价

    Examples
    --------
    >>> WINNER_COST = ts_WINNER_COST(HIGH, LOW, VWAP, turnover_ratio / 100, CLOSE, 0.5)

    Notes
    -----
    该函数仅对日线分析周期有效

    """
    dtype = Struct([Field(f"column_{i}", Float64) for i in range(2)])
    return struct([high, low, avg, turnover, close, cost]).map_batches(lambda xx: batches_i2_o2([xx.struct[i].to_numpy().astype(float) for i in range(6)], _WINNER_COST, step), return_dtype=dtype)
