# Point In Time

Essentially it is due to the fix of historical erroneous data. Sometimes we even need to fix the historical market data.

1. Market data：date\asset\features\[update_time]
2. Financial statement data：date\asset\features\update_time

`date`: time of the report,
`update_time`: time of the release

## Update of Financial Statement Data

There are two ways to download the data.

1. Download by `date`, just specify the 4 report periods of each year. But if the historical report period is changed in the new record, and the user cannot know which report period it is, the entire data needs to be downloaded.
2. Download by `update_time`, which can perfectly solve the above problem. But the release date is not fixed, and holidays may also occur. Downloading by day is inefficient, so it needs to be downloaded period by period.

## Storage of Financial Statement Data

Since the data is downloaded by `update_time`, it should be also stored by `update_time`, which is convenient for reading and updating.

## Storage of Historical Market Data

In market data, we do not need to distinguish the `date` and `update_time`.

Two ways to modify market data:

1. Similar to financial statement data, only add without modifying.
    - Stored by day, is the record appended to an extra file or added to the file of that day?
    - Because the storage is ordered by `update_time`, it should be added to the extra last file.
2. Directly modify the original data. In market data, the `update_time` is generally omitted, so the update method is more often to replace the old data.

## Dealing with data from PIT

1. The data update time may be on weekends. But the ideal way is to handle it the same as market data, so the `date` needs to be moved to Friday, and the `update_time` remains the same as the real time.
2. The PIT processing is divided into three steps.
  - filter out the corresponding time of the DF,
  - calculate the various time series indicators on each DF separately (or not),
  - take the latest part of each DF for merging.

After this, the time series calculation cannot be executed again, because it will introduce future data.


# Point In Time

本质上是因发现历史数据有问题，需要进行修改而产生的。我们使用的行情数据其实也会有修改需求。所以是否能将其一起讨论呢？

1. 行情数据：date\asset\features\[update_time]
2. 财务数据：date\asset\features\update_time

date表示报告期、update_time表示公布日期

## 财务数据的更新

1. 按报告期date下载，只要指定每年的4个报告期即可下载。但新记录中如果改了历史报告期。而用户又无法知道是哪个报告期，得全下
2. 按公布日update_time下载，能完美的解决上面的问题。但公布日不固定、节假日也会发生，按日下载又效率低，所以得按时间段下载

## 财务数据的存储

由于已经选定用update_time来下载数据，所以存储时也按update_time来分文件名，这样也方便读取和更新

## 行情数据的更新与存储

行情数据中，date其本质就是报告期，又因更新时间等于报告期，所以被省略了，同股票date一般也不会重复，如有重复，必有update_time字段做区分，否则数据有误。

行情数据有两种修改方法:

1. 仿财务数据，只添加不修改。（其实财务数据发布后在第二天交易开始前发生了更新可以直接修改）
    - 按日分文件存储，记录是添加到最后一个文件中，还是添加到那天的文件呢？
    - 因为存储是按update_time有序，所以应当添加到最后
2. 直接修改原数据。在行情数据由于一般省略了update_time，所以更新方式也更多是覆盖

## PIT数据的处理

1. 数据更新时间可能是周末。但理想方式是与行情一样处理，所以date需要移动到周五,update_time保持真实时间
2. PIT处理分三步，第一步过滤出对应时间的DF、第二步在每个DF上分别计算各时序指标（也可不算），第三步取每个DF中最新部分进行合并

不能在第三步后再执行时序计算了，因为这会引入未来数据