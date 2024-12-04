"""
通过`import`直接导入或更名的函数

```python
from polars_ta.wq.time_series import ts_corr as CORREL  # noqa
```

"""

from polars import Expr

from polars_ta.wq.time_series import ts_corr as CORREL  # noqa
from polars_ta.wq.time_series import ts_std_dev


def BETA(high: Expr, low: Expr, timeperiod: int = 5) -> Expr:
    raise


def LINEARREG(close: Expr, timeperiod: int = 14) -> Expr:
    raise


def LINEARREG_ANGLE(close: Expr, timeperiod: int = 14) -> Expr:
    raise


def LINEARREG_INTERCEPT(close: Expr, timeperiod: int = 14) -> Expr:
    raise


def LINEARREG_SLOPE(close: Expr, timeperiod: int = 14) -> Expr:
    raise


def STDDEV(close: Expr, timeperiod: int = 5, nbdev: float = 1) -> Expr:
    return ts_std_dev(close, timeperiod, ddof=0) * nbdev


def TSF(close: Expr, timeperiod: int = 14) -> Expr:
    raise


def VAR(close: Expr, timeperiod: int = 5, nbdev: float = 1) -> Expr:
    return close.rolling_var(timeperiod, ddof=0) * nbdev
