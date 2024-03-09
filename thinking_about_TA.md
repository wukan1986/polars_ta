# About Technical Indicators

There are three types of technical indicators

1. Time series indicators. Must be sorted by time, and then calculated. It is difficult to handle null in the middle.
2. Cross-sectional indicators. No order requirement, the whole row is calculated, and there may be a null, which needs to be compatible.
3. Single element indicators. Can be single-column, multi-column, and can be calculated in any way.

In terms of calculation, time series indicators have one more dimension than cross-sectional indicators. For example

1. `cs_rank` is a cross-sectional sort
2. `ts_rank` is to calculate `cs_rank` in a rolling time window, take `[-1]` each time, and then concatenate them

3. `cs_zscore` first aggregates to calculate `mean` and `std`, then broadcast `mean` and `std`, and perform a one-dimensional calculation with `x`

So common technical indicators are generally the flexible application of `aggregation` and `broadcasting`

Since `ts_` indicators are based on `cs_`, theoretically `rolling` can be used directly, but in practice, it is generally not used in this way.
`cs_` is generally rewritten in `cython`, `numba`, etc., and if you mix `ts_` and `cs_`, `ts_` will make thousands of `cs_` call.
So it is common to put `rolling` operations into `cython`, `numba`, etc. as well.

# 对技术指标的再思考

用使用方式来说，指标分为三种

1. 时序指标。必须按时间排序，然后计算，中段出现null比较难处理
2. 截面指标。对顺序无要求，整行计算即可，数据会有null，需要能兼容
3. 单元素指标。可单列、多列，可任意方式计算

从计算原理上来说，时序指标比截面指标多了一个计算维度。例如

1. `cs_rank`是横截面排序
2. 而`ts_rank`是对数据滚动时间窗口计算`cs_rank`,每次取`[-1]`，然后拼接起来

又例如

1. `cs_zscore`先聚合计算`mean`和`std`
2. 然后将`mean`与`std`广播,与`x`进行一维计算

所以常见的技术指标一般是`聚合`与`广播`的灵活应用

由于`ts_`指标基于`cs_`,理论上直接`rolling`即可，但在实践中一般不这么用，因为为了求快，`cs_`一般是更底层的语言编写，如`cython`、`numba`等，
本来`cs_`版`python`与底层只交互一次，而`ts_`调用`cs_`版会导致交互千万次，性能极低
一般是把`rolling`操作也放在底层，在底层循环调用`cs_`底层版

