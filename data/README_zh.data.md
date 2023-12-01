### 数据集说明
#### 数据特征说明
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
* 构建数据集EETh1_96_96.csv和EETh1_96_336.csv, 位于/data/train_data
* 其中EETh1_96_96.csv 的维度为(17229, 1344), EETh1_96_336.csv的维度为(16989, 3024)
* /data_processing_demo 提供了一个将数据集划分成训练集，验证集，测试集的样例
