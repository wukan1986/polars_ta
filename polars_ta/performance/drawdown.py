from polars import Expr


def ts_drawdown(close: Expr) -> Expr:
    return close.cum_max() - close
