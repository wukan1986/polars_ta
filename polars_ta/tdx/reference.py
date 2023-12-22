from polars_ta.ta.overlap import EMA
from polars_ta.ta.overlap import SMA as MA
from polars_ta.ta.volatility import TRANGE as TR
from polars_ta.wq.cross_sectional import rank as RANK
from polars_ta.wq.time_series import ts_count as COUNT
from polars_ta.wq.time_series import ts_decay_linear as WMA
from polars_ta.wq.time_series import ts_delay as REF
from polars_ta.wq.time_series import ts_delta as DIFF
from polars_ta.wq.time_series import ts_max as HHV
from polars_ta.wq.time_series import ts_min as LLV
from polars_ta.wq.time_series import ts_sum as SUM

EXPMA = EMA  # 别名

SMA = 0  # TODO 这里需要实测

# 防止IDE格式化时自动删除
_ = EXPMA, MA, TR
_ = RANK, COUNT, WMA, REF, DIFF, HHV, LLV, SUM
