"""
通过`import`直接导入或更名的函数

```python
from polars_ta.wq.time_series import ts_corr as CORREL  # noqa
```

"""

from polars import Expr

from polars_ta.wq.time_series import ts_corr as CORREL  # noqa
from polars_ta.wq.time_series import ts_decay_linear as _ts_decay_linear
from polars_ta.wq.time_series import ts_mean as _ts_mean
from polars_ta.wq.time_series import ts_std_dev


def BETA(high: Expr, low: Expr, timeperiod: int = 5) -> Expr:
    raise


def LINEARREG(close: Expr, timeperiod: int = 14) -> Expr:
    """Linear Regression

    Value at the last point of the least-squares line fitted to the last
    `timeperiod` values.

    Notes
    -----
    With x = 1..n over the window, the slope is `6 * (WMA - SMA) / (n - 1)`,
    so the end point `SMA + slope * (n - 1) / 2` equals `3 * WMA - 2 * SMA`.

    References
    ----------
    https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_LINEARREG.c

    """
    # The identity only holds on a full window, so MIN_SAMPLES must not shrink it
    wma = _ts_decay_linear(close, timeperiod, min_samples=timeperiod)
    sma = _ts_mean(close, timeperiod, min_samples=timeperiod)
    return wma * 3 - sma * 2


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
