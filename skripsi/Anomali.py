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
import pickle

regressor = pickle.load(open('../model/lstm.pkl', 'rb'))
sc = pickle.load(open('../model/scaler.pkl', 'rb'))

plt.style.use('fivethirtyeight')

TIMESTEPS = 60


# Some functions to help out with
def plot_predictions(test, predicted, anomaly):
    plt.plot(test[:, 0], color='c', label='Kelembaban')
    plt.scatter(anomaly[:, 0], anomaly[:, 1], color='red', label='Anomali', zorder=9)
    plt.plot(predicted[:, 0], color='g', label='Prediksi Kelembaban')
    plt.plot(test[:, 1], color='y', label='Suhu Tanah')
    plt.plot(predicted[:, 1], color='b', label='Prediksi Suhu Tanah')
    plt.plot(test[:, 2], color='black', label='Suhu Permukaan')
    plt.plot(predicted[:, 2], color='m', label='Prediksi Suhu Permukaan')
    plt.plot(test[:, 3], color='red', label='Curah')
    plt.plot(predicted[:, 3], color='k', label='Prediksi Curah')
    plt.title('Prediksi Sensor')
    plt.xlabel('Time')
    plt.ylabel('Nilai Sensor')
    plt.legend()
    plt.show()


# First, we get the data
test_set = pd.read_csv('../data/Data_Gabung_Detik.csv', index_col=False, parse_dates=['waktu'])
# test_set = test_set[['waktu', 'kelembaban', 'suhu', 'permukaan', 'curah']]
test_set = test_set[['waktu', 'kelembaban_2', 'suhu_2', 'suhu_permukaan', 'curah']]
test_set.set_index('waktu', inplace=True)
test_set = test_set['2020-03-10 23:59:59':'2020-03-11 23:59:59']

# Checking for missing values
test_set = test_set[:].values
test_set = test_set.reshape(-1, 4)
test_set_scaled = sc.transform(test_set)

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
y_test = sc.inverse_transform(y_test)

print(y_pred)
print(y_test)

anomali = []
THRESHOLD = 10
for i in range(len(y_pred)):
    for j in range(4):
        if (abs(y_pred[i][j] - y_test[i][j]) > THRESHOLD):
            anomali.append([i, y_test[i][j]])

anomali = np.array(anomali)
print(anomali)
# Visualizing the results for LSTM
plot_predictions(y_test, y_pred, anomali)
