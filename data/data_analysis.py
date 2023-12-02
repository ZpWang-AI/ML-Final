from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot as plt
import seaborn as sns

# load raw data
file = './raw_data/ETTh1.csv'
dataset = read_csv(file, header=0, index_col=0)

# times = [int(i[11:13]) for i in dataset.index]
# dataset.insert(7, 'time', times)

values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6]
i = 1
# plot each column
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:1000, group])
    plt.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
plt.show()

sns.heatmap(dataset.corr())
plt.show()