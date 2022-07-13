# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
# from keras.optimizers import SGD
# from tensorflow.keras.optimizers import SGD
# from sklearn.preprocessing._data import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt
from sklearn import preprocessing
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time

plt.style.use('fivethirtyeight')

TIMESTEPS = 60

# First, we get the data
dataset = pd.read_csv('../data/Data_Gabung_Detik.csv', index_col=False, parse_dates=['waktu'])
dataset = dataset[['waktu', 'kelembaban_2', 'suhu_2', 'suhu_permukaan', 'curah']]

print('test_set', dataset)
dataset['jam'] = dataset['waktu'].dt.hour
print('time',dataset)
dataset.set_index('waktu', inplace=True)
dataset = dataset['2020-04-16 12:00:00':'2020-04-16 23:00:00']
print(dataset.head())

#Preparing the data for LSTM model
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(dataset)
data_n = pd.DataFrame(np_scaled)
print(data_n)

#Important parameters and training/Test size
prediction_time = 1
testdatasize = 1000
unroll_length = 50
testdatacut = testdatasize + unroll_length  + 1

#Training data
x_train = data_n[0:-prediction_time-testdatacut].values
y_train = data_n[prediction_time:-testdatacut  ][0].values

#Test data
x_test = data_n[0-testdatacut:-prediction_time].values
y_test = data_n[prediction_time-testdatacut:  ][0].values

def unroll(data,sequence_length=24):
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)

#Adapt the datasets for the sequence data shape
x_train = unroll(x_train,unroll_length)
x_test  = unroll(x_test,unroll_length)
y_train = y_train[-x_train.shape[0]:]
y_test  = y_test[-x_test.shape[0]:]

#Shape of the data
print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)

#Building the model
model = Sequential()

model.add(LSTM(input_dim=x_train.shape[-1], units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('compilation time : {}'.format(time.time() - start))

model.fit(x_train, y_train, batch_size=3028, epochs=50, validation_split=0.1)

#Visualizing training and validaton loss
# plt.figure(figsize = (10, 5))
# plt.plot(model.history.history['loss'], label = 'Loss')
# plt.plot(model.history.history['val_loss'], Label = 'Val_Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.grid()
# plt.legend()

# import pickle
#
# with open('../model/lstm2.pkl', 'wb') as f:
#     pickle.dump(regressor, f)
#
# with open('../model/scaler2.pkl', 'wb') as f:
#     pickle.dump(sc, f)