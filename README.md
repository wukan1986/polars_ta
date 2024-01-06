# polars_ta

基于`polars`的算子库。实现量化投研中常用的技术指标、数据处理等函数。对于不易翻译成`Expr`的库（如：`TA-Lib`）也提供了函数式调用的封装

## 安装

### 在线安装

```commandline
pip install -i https://pypi.org/simple --upgrade polars_ta  # 官方源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade polars_ta  # 国内镜像源
```

### 源码安装

```commandline
git clone --depth=1 https://github.com/wukan1986/polars_ta.git
cd polars_ta
python -m build
cd dist
pip install polars_ta-0.1.2-py3-none-any.whl
```

## 使用方法

参考`examples`目录即可，例如：

```python
# 如果需要在`expr_codegen`中使用，需要有`ts_`等前权，这里导入提供了前缀
from polars_ta.prefix.tdx import *
# 导入wq公式
from polars_ta.prefix.wq import *

# 演示生成大量指标
df = df.with_columns([
   # 从wq中导入指标
   *[ts_returns(CLOSE, i).alias(f'ROCP_{i:03d}') for i in (1, 3, 5, 10, 20, 60, 120)],
   *[ts_mean(CLOSE, i).alias(f'SMA_{i:03d}') for i in (5, 10, 20, 60, 120)],
   *[ts_std_dev(CLOSE, i).alias(f'STD_{i:03d}') for i in (5, 10, 20, 60, 120)],
   *[ts_max(HIGH, i).alias(f'HHV_{i:03d}') for i in (5, 10, 20, 60, 120)],
   *[ts_min(LOW, i).alias(f'LLV_{i:03d}') for i in (5, 10, 20, 60, 120)],

   # 从tdx中导入指标
   *[ts_RSI(CLOSE, i).alias(f'RSI_{i:03d}') for i in (6, 12, 24)],
])
```

## 设计原则

1. 调用方法由`成员函数`换成`独立函数`。输入输出使用`Expr`，避免使用`Series`
2. 优先实现`wq`公式，它仿`WorldQuant Alpha`公式，与官网尽量保持一致。如果部分功能实现在此更合适将放在此处
3. 其次实现`ta`公式，它相当于`TA-Lib`的`polars`风格的版本。优先从`wq`中导入更名
4. 最后实现`tdx`公式，它也是优先从`wq`和`ta`中导入
5. `talib`的函数名与参数与原版`TA-Lib`完全一致
6. 如果出现了命名冲突，建议调用优先级为`wq`、`ta`、`tdx`、`talib`。因为优先级越高，实现方案越接近于`Expr`

## 指标区别

请参考[compare](compare.md)

## 空值处理

请参考[nan_to_null](nan_to_null.md)

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
    - 实现代码[codegen_talib2.py](tools/codegen_talib2.py)
    - 生成结果[\_\_init\_\_.py](polars_ta/talib/__init__.py)
    - 使用演示[demo_ta3.py](examples/demo_ta3.py)
    - 优点：即可以输入到遗传算法，`IDE`还有智能提示

## 开发调试

```commandline
git clone --depth=1 https://github.com/wukan1986/polars_ta.git
cd polars_ta
pip install -e .
```

注意：如果你在`ta`或`tdx`中添加了新的函数，请再运行`tools`下的`prefix_ta.py`或`prefix_tdx.py`，用于生成对应的前缀文件。前缀文件方便在`expr_codegen`中使用

## 参考

- https://github.com/pola-rs/polars
- https://github.com/TA-Lib/ta-lib
- https://github.com/twopirllc/pandas-ta
- https://github.com/bukosabino/ta
- https://github.com/wukan1986/ta_cn

