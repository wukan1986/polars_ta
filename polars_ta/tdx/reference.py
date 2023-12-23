import polars as pl

from polars_ta.ta.overlap import EMA
from polars_ta.ta.overlap import SMA as MA
from polars_ta.ta.volatility import TRANGE as TR
# from polars_ta.wq.arithmetic import reverse as REVERSE
from polars_ta.wq.cross_sectional import rank as RANK
from polars_ta.wq.time_series import ts_arg_max as HHVBARS
from polars_ta.wq.time_series import ts_arg_min as LLVBARS
from polars_ta.wq.time_series import ts_count as COUNT
from polars_ta.wq.time_series import ts_decay_linear as WMA
from polars_ta.wq.time_series import ts_delay as REF
from polars_ta.wq.time_series import ts_delta as DIFF
from polars_ta.wq.time_series import ts_max as HHV
from polars_ta.wq.time_series import ts_min as LLV
from polars_ta.wq.time_series import ts_product as MULAR
from polars_ta.wq.time_series import ts_sum as SUM

EXPMA = EMA  # 别名

SMA = 0  # TODO 这里需要实测

# 防止IDE格式化时自动删除
_ = EXPMA, MA, TR
_ = RANK, HHVBARS, LLVBARS, COUNT, WMA, REF, DIFF, HHV, LLV, MULAR, SUM
_ = MULAR


def RANGE(a: pl.Expr, b: pl.Expr, c: pl.Expr) -> pl.Expr:
    """A在B和C范围之间,B<A<C."""
    return (b < a) & (a < c)
