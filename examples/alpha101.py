from polars_ta.imports.expr import *

# 因子定义
OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT, = pl.col('OPEN'), pl.col('HIGH'), pl.col('LOW'), pl.col('CLOSE'), pl.col('VOLUME'), pl.col('AMOUNT'),
RETURNS, VWAP, CAP, = pl.col('RETURNS'), pl.col('VWAP'), pl.col('CAP'),
ADV5, ADV10, ADV15, ADV20, ADV30, ADV40, ADV50, ADV60, ADV81, ADV120, ADV150, ADV180, = (
    pl.col('ADV5'), pl.col('ADV10'), pl.col('ADV15'), pl.col('ADV20'),
    pl.col('ADV30'), pl.col('ADV40'), pl.col('ADV50'), pl.col('ADV60'),
    pl.col('ADV81'), pl.col('ADV120'), pl.col('ADV150'), pl.col('ADV180'),)
SECTOR, INDUSTRY, SUBINDUSTRY, = pl.col('SECTOR'), pl.col('INDUSTRY'), pl.col('SUBINDUSTRY'),

# 列因子才可以直拉调用，而ts_、cs_和gp_等公式需要套用group_by，请使用expr_codegen项目
alpha_041 = (((HIGH * LOW) ** 0.5) - VWAP)
alpha_054 = ((-1 * ((LOW - CLOSE) * (OPEN ** 5))) / ((LOW - HIGH) * (CLOSE ** 5)))
alpha_101 = ((CLOSE - OPEN) / ((HIGH - LOW) + 0.001))

# 测试数据
df = pl.DataFrame(
    {
        "OPEN": np.random.rand(1000),
        "HIGH": np.random.rand(1000),
        "LOW": np.random.rand(1000),
        "CLOSE": np.random.rand(1000),
        "VWAP": np.random.rand(1000),
    }
)

# 计算
df = df.with_columns(
    alpha_041=alpha_041,
    alpha_054=alpha_054,
    alpha_101=alpha_101,
)

print(df)
