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

## Evolve of Our TA-Lib Wrappers

1. `Expr.map_batches` can be used to call third-party libraries, such as `TA-Lib, bottleneck`. But because of the input and output format requirements, you need to wrap the third-party API with a function.
   - Both input and output can only be one column. If you want to support multiple columns, you need to convert them to `pl.Struct`. After that, you need to use `unnest` to split `pl.Struct`.
   - The output must be `pl.Series`

2. Start to use `register_expr_namespace` to simplify the code
   - Implementation [helper.py](polars_ta/utils/helper.py)
   - Usage demo [demo_ta1.py](examples/demo_ta1.py)
   - Pros: Easy to use
   - Cons:
       - The `member function call mode` is not convenient for inputting into genetic algorithms for factor mining
       - `__getattribute__` dynamic method call is very flexible, but loses `IDE` support.

3. Prefix expression. Convert all member functions into formulas
   - Implementation [wrapper.py](polars_ta/utils/wrapper.py)
   - Usage demo [demo_ta2.py](examples/demo_ta2.py)
   - Pros: Can be input into our implementation of genetic algorithms
   - Cons: `__getattribute__` dynamic method call is very flexible, but loses `IDE` support.

4. Code generation.
   - Implementation [codegen_talib.py](tools/codegen_talib.py)
   - Generated result will be at [\_\_init\_\_.py](polars_ta/talib/__init__.py)
   - Usage demo [demo_ta3.py](examples/demo_ta3.py)
   - Pros:
       - Can be input into our implementation of genetic algorithms
       - `IDE` support

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

## TA-Lib封装的演化

1. `Expr.map_batches`可以实现调用第三方库，如`TA-Lib, bottleneck`。但因为对输入与输出格式有要求，所以还需要用函数对第三方API封装一下。
    - 输入输出都只能是一列，如要支持多列需转换成`pl.Struct`。事后`pl.Struct`要拆分需使用`unnest`
    - 输出必须是`pl.Series`
2. 参数多，代码长。开始使用`register_expr_namespace`来简化代码
    - 实现代码[helper.py](polars_ta/utils/helper.py)
    - 使用演示[demo_ta1.py](examples/demo_ta1.py)
    - 优点：使用简单
    - 不足：`成员函数调用模式`不便于输入到遗传算法中进行因子挖掘
    - 不足：`__getattribute__`动态方法调用非常灵活，但失去了`IDE`智能提示
3. 前缀表达式。将所有的成员函数都转换成公式
    - 实现代码[wrapper.py](polars_ta/utils/wrapper.py)
    - 使用演示[demo_ta2.py](examples/demo_ta2.py)
    - 优点：可以输入到遗传算法
    - 不足：`__getattribute__`动态方法调用非常灵活，但失去了`IDE`智能提示
4. 代码自动生成。
    - 实现代码[codegen_talib.py](tools/codegen_talib.py)
    - 生成结果[\_\_init\_\_.py](polars_ta/talib/__init__.py)
    - 使用演示[demo_ta3.py](examples/demo_ta3.py)
    - 优点：即可以输入到遗传算法，`IDE`还有智能提示

