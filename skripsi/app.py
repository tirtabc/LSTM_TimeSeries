# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
# from keras.optimizers import SGD
# from tensorflow.keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
import pickle

regressor = pickle.load(open('../model/lstm.pkl', 'rb'))
sc = pickle.load(open('../model/scaler.pkl', 'rb'))
y_pred = regressor.predict([[[[80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4],
                         [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4],
                         [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4],
                         [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4],
                         [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4],
                         [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4],
                         [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4],
                         [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4],
                         [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4],
                         [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4],
                         [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4],
                         [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4], [80, 80, 80, 4]]]])

print(y_pred)

plt.style.use('fivethirtyeight')

TIMESTEPS = 60


# Some functions to help out with
def plot_predictions(test, predicted):
    plt.plot(test[:, 0], color='red', label='Kelembaban')
    plt.plot(predicted[:, 0], color='g', label='Prediksi Kelembaban')
    plt.plot(test[:, 1], color='y', label='Suhu Tanah')
    plt.plot(predicted[:, 1], color='b', label='Prediksi Suhu Tanah')
    plt.plot(test[:, 2], color='red', label='Suhu Permukaan')
    plt.plot(predicted[:, 2], color='m', label='Prediksi Suhu Permukaan')
    plt.plot(test[:, 3], color='red', label='Curah')
    plt.plot(predicted[:, 3], color='k', label='Prediksi Curah')
    plt.title('Prediksi Sensor')
    plt.xlabel('Time')
    plt.ylabel('Nilai Sensor')
    plt.legend()
    plt.show()


def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))


# First, we get the data
import pandas as pd

TIMESTEPS = 60
dataset1 = pd.read_csv('../data/data_09_10.csv', index_col=False, parse_dates=['waktu'])
dataset = dataset1[['waktu', 'kelembaban', 'suhu', 'permukaan', 'curah']]
dataset.set_index('waktu', inplace=True)

# Checking for missing values
y_test = dataset['2020-03-10':].values
print('y_test', y_test.shape)

inputs = dataset[len(dataset) - len(y_test) - TIMESTEPS:].values
inputs = inputs.reshape(-1, 4)
inputs = sc.transform(inputs)
print('y_test', inputs.shape)

# Preparing X_test and predicting the prices
X_test = []
size = len(inputs)
for i in range(TIMESTEPS, size):
    X_test.append(inputs[i - TIMESTEPS:i])

X_test = np.array(X_test)
print('y_test', X_test.shape)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))
y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)



# Evaluating our model
return_rmse(y_test, y_pred)

# Visualizing the results for LSTM
plot_predictions(y_test, y_pred)

testdatasize = 1000
outliers_fraction = 0.01

loaded_model = regressor
diff = []
ratio = []
#
p = loaded_model.predict(X_test)
for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))

#
# #Plotting the prediction and the reality (for the test data)
# plt.figure(figsize = (10, 5))
# plt.plot(p,color='red', label='Prediction')
# plt.plot(y_test,color='blue', label='Test Data')
# plt.legend(loc='upper left')
# plt.grid()
# plt.legend()
# plt.show()

# Pick the most distant prediction/reality data points as anomalies
diff = pd.Series(diff)

number_of_outliers = int(outliers_fraction * len(diff))
print(number_of_outliers)
threshold = diff.nlargest(number_of_outliers).min()
# Data with anomaly label
test = (diff >= threshold).astype(int)
complement = pd.Series(0, index=np.arange(len(dataset) - testdatasize))
dataset1['anomaly27'] = complement.append(test, ignore_index='True')

print(dataset1['anomaly27'].value_counts())
print(dataset1)

# Visualizing anomalies (Red Dots)
# plt.figure(figsize=(15,10))
# a = dataset1.loc[dataset1['anomaly27'] == 1, ['time_epoch', 'value']] #anomaly
# plt.plot(dataset1['time_epoch'], dataset1['value'], color='blue')
# plt.scatter(a['time_epoch'],a['value'], color='red', label = 'Anomaly')
# plt.axis([1.370*1e7, 1.405*1e7, 15,30])
# plt.grid()
# plt.legend()
# plt.show()
