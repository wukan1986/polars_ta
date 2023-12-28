from polars import Expr


def drawdown(close: Expr) -> Expr:
    return close.cum_max() - close
