from polars import Expr


def ts_max_drawdown(close: Expr) -> Expr:
    """最大回撤"""
    return close - close.cum_max()


def ts_max_drawdown_rate(close: Expr) -> Expr:
    """最大回撤率"""
    return close / close.cum_max() - 1
