# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error

plt.style.use('fivethirtyeight')

TIMESTEPS = 60

# First, we get the data
dataset = pd.read_csv('../data/Data_Gabung_Detik.csv', index_col=False, parse_dates=['waktu'])
dataset = dataset[['waktu', 'kelembaban_2', 'suhu_2', 'suhu_permukaan', 'curah']]
dataset.set_index('waktu', inplace=True)
print(dataset.head())


def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))


# Checking for missing values
training_set = dataset[:'2020-03-09'].values
test_set = dataset['2020-03-10':'2020-03-12'].values

# Scaling the training set
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

# Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output
# So for each element of training set, we have 60 previous training set elements
X_train = []
y_train = []
size = len(training_set)
for i in range(TIMESTEPS, size):
    X_train.append(training_set_scaled[i - TIMESTEPS:i])
    y_train.append(training_set_scaled[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))

# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 4)))
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
regressor.add(Dense(units=4))

# Compiling the RNN
regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
# Fitting to the training set
regressor.fit(X_train, y_train, epochs=50, batch_size=32)

# Testing
test_set_scaled = dataset[len(dataset) - len(test_set) - TIMESTEPS:].values
test_set_scaled = test_set_scaled.reshape(-1, 4)
test_set_scaled = sc.transform(test_set_scaled)

# Preparing X_test and predicting the prices
X_test = []
y_test = []
size = len(test_set_scaled)
for i in range(TIMESTEPS, size):
    X_test.append(test_set_scaled[i - TIMESTEPS:i])
    y_test.append(test_set_scaled[i])
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshaping X_train for efficient modelling
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))

y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)

# Evaluating our model
return_rmse(y_test, y_pred)

# Output Model
import pickle

with open('../model/lstm.pkl', 'wb') as f:
    pickle.dump(regressor, f)

with open('../model/scaler.pkl', 'wb') as f:
    pickle.dump(sc, f)
