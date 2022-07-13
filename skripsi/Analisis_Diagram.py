# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

# First, we get the data
dataset = pd.read_csv('../data/Data_Gabung_Detik.csv', index_col=False, parse_dates=['waktu'])
# dataset = dataset[['waktu', 'kelembaban_2', 'suhu_2', 'suhu_permukaan', 'curah']]
dataset = dataset[['waktu', 'kelembaban_2']]
dataset.set_index('waktu', inplace=True)
print(dataset.head(200))

# Checking for missing values
training_set = dataset['2020-04-12':'2020-04-16']
test_set = dataset['2020-04-16':'2020-04-17']

# dataset1 = dataset['kelembaban_2'].values
# dataset1 = dataset. astype(int)
# dataset['kelembaban_2'] = dataset1


training_set['kelembaban_2'].plot()
# training_set['kelembaban_2'].hist()
plt.show()