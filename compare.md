# Differences in Indicators

## 1. EMA alike indicators

### 1.1 EMA

1. EMA(CLOSE, 10), `talib.set_compatibility(0)` is the default, and is equivalent to`EXPMEMA`
    - The first not nan value starts as `talib.SMA(CLOSE, 10)`
2. EMA(CLOSE, 10), `talib.set_compatibility(1)`
    - The first not nan value starts as `CLOSE`

Since the logic of `EMA` calculation has changed in `TA-Lib`'s `compatibility mode 0`, it is complex and inefficient to implement it using expressions.

We only implement `compatibility mode 1`.
It just so happens that the Chinese stock software only implements `compatibility mode 1`.
You can compare the full data with `TA-Lib` for unit testing.

Indicators affected by `EMA` include `MACD, DEMA, TEMA, TRIX, T3`, etc.

### 1.2 Chinese Version of SMA(X, N, M)

In essence, it is `RMA compatibility mode 0`, that is,
the first valid value is the moving average. And then the difference is `alpha`

1. `SMA(X, N, M) = X.ema_mean(alpha=M/N)`
2. `RMA(X, N) = X.ema_mean(alpha=1/N) = SMA(X, N, 1)`
3. `EMA(x, N) = X.ema_mean(alpha=2/(N+1)) = X.ema_mean(span=N)`

Refer to: [ewm_mean](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.ewm_mean.html#polars.Expr.ewm_mean)

In this case, we use `RMA compatibility mode 1`.
There exists error in switching between the two modes.
So please provide enough length of data.
The data in the later part can be used for unit testing.

Indicators affected by `SMA` include `ATR, RSI`, etc.


### 1.3 Moving Sum

Some indicators including `ADX` require the first value as a `SUM` rather than `SMA`. We will implement them later by `ema_mean(alpha=1/N)`


## 2. MAX/MIN

1. In package `wq`, `max_/min_` are cross-sectional operators, `ts_max/ts_min` are time series indicators
2. In `talib`, `MAX/MIN` are time series indicators, without cross-sectional operators.
3. In `ta`, to mimic `talib`, `MAX/MIN` are time series indicators, without cross-sectional operators.
4. In `tdx`, `MAX/MIN` are cross-sectional operators, `HHV/LLV` are time series indicators.


# 指标区别

## 1. EMA系列

### 1.1 EMA指标

1. EMA(CLOSE, 10)，`talib.set_compatibility(0)`，此为默认设置，等价于`EXPMEMA`
    - 第一个有效值为`talib.SMA(CLOSE, 10)`
2. EMA(CLOSE, 10)，`talib.set_compatibility(1)`
    - 第一个有效值为`CLOSE`

由于`TA-Lib`的`兼容模式0`在`EMA`计算时逻辑发生了变动，用表达式实现起来复杂、计算效率低。
所以本库只实现`兼容模式1`。正好国内股票软件其实也只实现了`兼容模式1`。可以全量数据与`TA-Lib`进行单元测试比较

因`EMA`受影响的指标有`MACD, DEMA, TEMA, TRIX, T3`等。

### 1.2 中国版SMA(X, N, M)

本质上是国外的`RMA 兼容模式0`，即第一个有效值为移动平均，然后就是`alpha`区别

1. `SMA(X, N, M) = X.ema_mean(alpha=M/N)`
2. `RMA(X, N) = X.ema_mean(alpha=1/N) = SMA(X, N, 1)`
3. `EMA(x, N) = X.ema_mean(alpha=2/(N+1)) = X.ema_mean(span=N)`

换算关系可参考: [ewm_mean](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.ewm_mean.html#polars.Expr.ewm_mean)

遇到这种情况，本项目还是用`RMA 兼容模式1`来代替，数据误差由大到小，所以请预先提供一定长度的数据。后面一段的数据可以单元测试

受影响的的指标有`ATR, RSI`等

### 1.3 移动求和

`ADX`等一类的指标第一个有效值算法为`SUM`，而不是`SMA`，之后使用`ema_mean(alpha=1/N)`。此类暂不实现

## 2. MAX/MIN等指标

1. 在`wq`中，`max_/min_`横向算子，`ts_max/ts_min`时序指标
2. 在`talib`中, `MAX/MIN`时序指标，没有横向算子
3. 在`ta`中，由于要模仿`talib`，所以有`MAX/MIN`时序指标，也没有横向算子
4. 在`tdx`中，`MAX/MIN`横向算子，`HHV/LLV`时序指标

