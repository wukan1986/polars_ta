from polars import Expr

from polars_ta.wq.time_series import ts_returns, ts_log_diff

# 对数收益
log_return = ts_log_diff
# 简单收益
percent_return = ts_returns


def cum_return(close: Expr) -> Expr:
    """累计收益"""

    return close / close.drop_nulls().first()
