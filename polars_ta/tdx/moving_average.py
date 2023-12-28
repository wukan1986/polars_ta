from polars import Expr

from polars_ta.ta.price import AVGPRICE
from polars_ta.tdx.reference import MA


def BBI(CLOSE: Expr, M1: int = 3, M2: int = 6, M3: int = 12, M4: int = 20) -> Expr:
    """
    BBI:(MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4;

    """
    return AVGPRICE(MA(CLOSE, M1), MA(CLOSE, M2), MA(CLOSE, M3), MA(CLOSE, M4))
