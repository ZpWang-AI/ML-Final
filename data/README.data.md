### 数据集说明
#### 数据特征说明
* time: 表示当前时间，由date得到
* HUFL: High UseFul Load
* HULL: High UseLess Load
* MUFL: Middle UseFul Load
* MULL: Middle UseLess Load
* LUFL: Low UseFul Load
* LULL: Low UseLess Load
* OT: Oil Temperature (target)

#### 数据来源
* 数据来自中国一个省的两个地区两年的数据，根据要求本次作业使用ETT-small-h1数据集
* ETT-small-h1 每隔1个小时一个数据 共2 year * 365 days * 24 hours = 17520个数据


#### 数据集构建
* 根据要求，根据过去96小时的全部数据预测将来96和336小时的全部数据
* 构建数据集data_96-96和data_96-336, 分别包含训练集，验证集和测试集
* /creat_dataset.py 提供了划分数据集的代码
* /data_processing_demo.py 提供了一个将数据集划分成训练集，验证集，测试集的样例
