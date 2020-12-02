# Importing the libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error

# TODO: Hapus timesteps2
TIMESTEPS = 3

# First, we get the data
dataset = pd.read_csv('../data/data_09_10.csv', index_col=False, parse_dates=['waktu'])
dataset = dataset[['waktu', 'kelembaban', 'suhu', 'permukaan', 'curah']]
dataset.set_index('waktu', inplace=True)

print(dataset.head())

# Checking for missing values
training_set = dataset.iloc[:, 0:4].values
# Kelembaban
# test_set = dataset.iloc[:, 0:1].values

# We have chosen 'High' attribute for prices. Let's see what it looks like
# dataset["kelembaban"].plot()
# plt.title('kelemababn')
# plt.show()

# Scaling the training set
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output
# So for each element of training set, we have 60 previous training set elements
X_train = []
y_train = []
size = len(dataset)
for i in range(TIMESTEPS, size):
    X_train.append(training_set_scaled[i - TIMESTEPS:i, 1:4])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 3))

# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 3)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
# Fitting to the training set
regressor.fit(X_train, y_train, epochs=3, batch_size=32)

# Export model
with open('../model/lstm.pkl', 'wb') as f:
    pickle.dump(regressor, f)
