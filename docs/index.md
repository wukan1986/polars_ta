# polars_ta

尽量接近`WorldQuant Alpha101`风格函数，但部分又做了一定的调整

例如：

1. x.abs().log()
    - 可利用`IDE`的自动补全，输入方便
    - 默认是一个输入，多输入要通过参数列表。输入不统一
2. log(abs_(x))
    - 都是通过参数列表，输入统一
    - 一层套一层，正好对应表达式树，直接可用于遗传规划
    - `abs`与`python`内置函数冲突，使用`abs_`代替,同样还有`and_`、`int_`、`max_`等

## References

- https://platform.worldquantbrain.com/learn/operators/operators
- https://github.com/TA-Lib/ta-lib