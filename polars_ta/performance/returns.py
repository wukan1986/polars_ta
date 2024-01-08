from polars import Expr

# 对数收益
from polars_ta.wq.time_series import ts_log_diff as ts_log_return  # noqa
# 简单收益
from polars_ta.wq.time_series import ts_returns as ts_percent_return  # noqa


def ts_cum_return(close: Expr) -> Expr:
    """累计收益"""

    return close / close.drop_nulls().first()
