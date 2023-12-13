# polars_ta

## 设计原则

1. 输入数据使用`Expr`，少用`Series`
2. 函数命名优先向`TA-Lib`靠拢，然后向`WorldQuant`靠拢，最后才仿国内通达信等表达式
3. 计算速度优先
4. 标准化优先。与其它系统结果不一致时需示明区别

## 指标区别

### EMA指标

1. EMA(CLOSE, 10)，talib.set_compatibility(0)，此为默认设置
    - 第一个有效值为talib.SMA(CLOSE, 10)
2. EMA(CLOSE, 10)，talib.set_compatibility(1)
    - 第一个有效值为CLOSE

由于`TA-Lib`的`兼容模式0`在`EMA`计算时逻辑发生了变动，用表达式实现起来复杂、计算效率低。
所以本库只实现`兼容模式1`。正好国内股票软件其实也只实现了`兼容模式1`。可以全量数据与`TA-Lib`进行单元测试比较

因`EMA`受影响的指标有`MACD, DEMA, TEMA, TRIX, T3`等。

### 中国版SMA(X, M, N)

本质上是国外的`RMA 兼容模式0`，即第一个有效值为移动平均

1. RMA(X, timeperiod)=X.ema_mean(alpha=1/timeperiod)
2. SMA(X, N, M)=X.ema_mean(alpha=M/N)

`alpha`与`span`的换算关系可参考`https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.ewm_mean.html#polars.Expr.ewm_mean`

遇到这种情况，本项目还是用`RMA 兼容模式1`来代替，数据误差由大到小，所以请预先提供一定长度的数据

受影响的的指标有`ATR, RSI`等

### 移动求和

`ADX`等一类的指标第一个有效值为SUM，而不是SMA，之后使用ema_mean(alpha=1/N)。此类暂不实现

## 参考

https://github.com/TA-Lib/ta-lib
https://github.com/twopirllc/pandas-ta
https://github.com/bukosabino/ta

