import polars as pl

from polars_ta.wq.arithmetic import log_diff
from polars_ta.wq.time_series import ts_returns

# 对数收益
log_return = log_diff
# 简单收益
percent_return = ts_returns


def cum_return(close: pl.Expr) -> pl.Expr:
    """累计收益"""

    return close / close.drop_nulls().first()
