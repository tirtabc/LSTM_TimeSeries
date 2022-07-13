# Importing the libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
# from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error

TIMESTEPS = 60

# First, we get the data
dataset = pd.read_csv('../data/data_10.csv', index_col=False, parse_dates=['waktu'])
dataset = dataset[['waktu', 'kelembaban', 'suhu', 'permukaan', 'curah']]
dataset.set_index('waktu', inplace=True)
print(dataset.head())

# Now to get the test set ready in a similar way as the training set.
# The following has been done so first 60 entires of test set have 60 previous values which is impossible to get unless we take the whole
# 'High' attribute data for processing
dataset_total = pd.concat((dataset["High"][:'2016'],dataset["High"]['2017':]),axis=0)
inputs = dataset_total[len(dataset_total)-len(test_set) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.transform(inputs)

# Preparing X_test and predicting the prices
X_test = []
for i in range(60,311):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
print('Data test',X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
print('Data test reshape',X_test)
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Evaluating our model
rmse = math.sqrt(mean_squared_error(test_set, predicted_stock_price))
print("The root mean squared error is {}.".format(rmse))
