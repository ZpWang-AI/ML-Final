from pandas import read_csv
from pandas import DataFrame
from pandas import concat

num_features = 8


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load raw data
file = './raw_data/ETTh1.csv'
dataset = read_csv(file, header=0, index_col=0)

# create column time
times = [int(i[11:13]) for i in dataset.index]
dataset.insert(7, 'time', times)

print(dataset.head(5))
values = dataset.values
# ensure all data is float
values = values.astype('float32')

# split dataset
train_num = int(0.6*len(values))
test_num = int(0.2*len(values))
train_value = values[:train_num, :]
val_value = values[train_num:train_num+test_num, :]
test_value = values[train_num+test_num:, :]

type2value = {'train': train_value, 'test': test_value, 'dev': val_value}
n_in = 96
for n_out in [96, 336]:
    for t in ['train', 'dev', 'test']:
        value = type2value[t]
        reframed = series_to_supervised(value, n_in, n_out)
        out_file = './data/data_{0}-{1}/{2}.csv'.format(n_in, n_out, t)
        reframed.to_csv(out_file)
        print('done')