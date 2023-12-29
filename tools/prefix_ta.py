from tools.prefix import codegen_import, save

lines = []
lines += codegen_import('polars_ta.ta.momentum', include_parameter=['timeperiod', 'fastperiod', 'fastk_period'])
lines += codegen_import('polars_ta.ta.overlap', include_modules=['polars_ta.wq.time_series'], include_func=['SMA', 'WMA'], include_parameter=['timeperiod'])
lines += codegen_import('polars_ta.ta.price', include_parameter=['timeperiod'])
lines += codegen_import('polars_ta.ta.statistic', include_parameter=['timeperiod'])
lines += codegen_import('polars_ta.ta.transform', include_modules=['polars_ta.wq.arithmetic'], include_parameter=['timeperiod'])
lines += codegen_import('polars_ta.ta.volatility', include_func=['TRANGE'], include_parameter=['timeperiod'])
lines += codegen_import('polars_ta.ta.volume', include_func=['AD', 'OBV'], include_parameter=['fastperiod'])
save(lines, module='polars_ta.prefix.ta', write=True)
