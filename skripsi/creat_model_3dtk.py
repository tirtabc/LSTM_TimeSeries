import numpy as np
import tensorflow as tf
import sys
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from pandas.plotting import register_matplotlib_converters
from keras.layers import Dense, LSTM, Dropout
from keras import regularizers
import scipy.stats as stats
import math


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

df = pd.read_csv(
    "../data/B5_1-3_3dtk.csv",
    parse_dates=['waktu'],
    index_col="waktu",
)

print(df)

# Select observations between two datetimes
# df = df.loc['2020-04-15 00:00:00':'2020-04-15 02:00:00']

print(df.shape)
print(df.head())

x_simple = np.array(df['kelembaban_tanah'])
y_simple = np.array(df['suhu_2'])
my_rho = np.corrcoef(x_simple, y_simple)
T_simple = np.array(df['kelembaban_tanah'])
Z_simple = np.array(df['suhu_permukaan'])
my_rho2 = np.corrcoef(T_simple, Z_simple)
print('koefisien',my_rho)
print('koefisien2',my_rho2)

# df['second'] = df.index.second
# df['minutes'] = df.index.min
# df['hour'] = df.index.hour
# df['day_of_month'] = df.index.day
# df['day_of_week'] = df.index.dayofweek
# df['month'] = df.index.month

from sklearn.preprocessing import RobustScaler, MinMaxScaler

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))
f_columns = ['suhu_2', 'suhu_permukaan']

# f_transformer = RobustScaler()
# kelembaban_tanah_transformer = RobustScaler()
f_transformer = MinMaxScaler()
kelembaban_tanah_transformer = MinMaxScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
print('f_transformer', f_transformer)
kelembaban_tanah_transformer = kelembaban_tanah_transformer.fit(train[['kelembaban_tanah']])
print('kelembaban_tanah_transformer', kelembaban_tanah_transformer)

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
print('train.loc[:, f_columns]', train.loc[:, f_columns])
train['kelembaban_tanah'] = kelembaban_tanah_transformer.transform(train[['kelembaban_tanah']])
print(train['kelembaban_tanah'])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['kelembaban_tanah'] = kelembaban_tanah_transformer.transform(test[['kelembaban_tanah']])

time_steps = 10

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train, train.kelembaban_tanah, time_steps)
X_test, y_test = create_dataset(test, test.kelembaban_tanah, time_steps)

print(X_train.shape, y_train.shape)
print('X_train :', X_train, 'y_train :', y_train)

model = Sequential()
model.add(LSTM(62, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]),
               kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])
# print('compilation time : {}'.format(time.time() - start))

model.fit(X_train, y_train, epochs=40, verbose=2, validation_split=0.1, batch_size=32, shuffle=False)

# plt.plot(model.history.history['loss'], label='train')
# plt.plot(model.history.history['val_loss'], label='test')
# plt.legend()
# plt.clf()

y_pred = model.predict(X_test)

# Accuracy Bukan untuk mengukur timeseries
# from sklearn.metrics import accuracy_score

# train_preds = np.where(model.predict(X_train) > 0.5, 1, 0)
# test_preds = np.where(model.predict(X_test) > 0.5, 1, 0)

# train_accuracy = accuracy_score(y_train, train_preds)
# test_accuracy = accuracy_score(y_test, test_preds)

# print(f'Train Accuracy : {train_accuracy:.4f}')
# print(f'Test Accuracy  : {test_accuracy:.4f}')

y_train_inv = kelembaban_tanah_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = kelembaban_tanah_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = kelembaban_tanah_transformer.inverse_transform(y_pred)

print('y_test_inv', y_test_inv, 'len y_test_inv ', len(y_test_inv.ravel()))
print('y_pred_inv', y_pred_inv, 'len y_pred_inv ', len(y_pred_inv.ravel()))
# estimate stdev of yhat
sum_errs = np.sum((y_test_inv.ravel() - y_pred_inv.ravel()) ** 2)
print('sum_errs', sum_errs)
stdev = np.sqrt(1 / (len(y_test_inv.ravel()) - 2) * sum_errs)
print('stdev', stdev)
# calculate prediction interval
interval = 1.96 * stdev
print('Prediction Interval: %.3f' % interval)


# def threshold(x_pred1):
#     xbar_sample = x_pred1.mean()
#     # print("Xbar_sample = \n", xbar_sample)
#     sigma_sample = x_pred1.std()
#     # print("Sigma sample = \n", sigma_sample)
#     SE_sample = sigma_sample / math.sqrt(data_bahan.size)
#     # print("SE_sample = \n", SE_sample)
#     # z score = mean+- 1.96 σ/akar dari n
#     z_critical = stats.norm.ppf(q=0.95)
#     # print("Z_Critical = \n", z_critical)
#     # bb = xbar_sample - z_critical * SE_sample
#     bb = xbar_sample - z_critical * SE_sample
#     bb = math.floor(bb)
#     # ba = xbar_sample + z_critical * SE_sample
#     ba = xbar_sample + z_critical * SE_sample
#     ba = math.ceil(ba)
#     return ba, bb
#
# bound_kel = list(threshold(bound_1[-5:]))
# print('Bound atas, bound bawah', bound_kel, bound_kel[0], bound_kel[1])

# calculate MSE
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# train_pred = model.predict(X_train)
# R_2_train = r2_score(y_train, train_pred)
# print('Training Score: %.2f R2 Score: ' % (R_2_train))

MSE = mean_squared_error(y_test, y_pred)
RMSE = sqrt(mean_squared_error(y_test, y_pred))
R_2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
print('Test Score: %.2f R2 Score: ' % (R_2))
print('Test Score: %.2f MAE: ' % (MAE))
print('Test Score: %.2f MSE' % (MSE))
print('Test Score: %.2f RMSE' % (RMSE))

# Predict With Test Data
# plt.plot(y_test_inv.flatten(), marker='.', label="true")
# plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
# plt.ylabel('Kelembamban')
# plt.xlabel('Time Step')
# plt.legend()
# plt.show();

# Predict With Future
plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Kelembamban')
plt.xlabel('Time Step')
plt.legend()
plt.show()

# Output Model
import pickle
#
# with open('../model/lstm_B5_1-3b_3dtk.pkl', 'wb') as f:
#     pickle.dump(model, f)
#
# with open('../model/scaler_B5_1-3b_3dtk.pkl', 'wb') as f:
#     pickle.dump(f_transformer, f)
#
# with open('../model/scaler2_B5_1-3b_3dtk.pkl', 'wb') as f:
#     pickle.dump(kelembaban_tanah_transformer, f)

with open('../model/lstm_Korelasi_kel-suhu.pkl', 'wb') as f:
    pickle.dump(my_rho, f)

with open('../model/lstm_Korelasi_kel-permukaan.pkl', 'wb') as f:
    pickle.dump(my_rho2, f)

with open('../model/lstm_interval.pkl', 'wb') as f:
    pickle.dump(interval, f)