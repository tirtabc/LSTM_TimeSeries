# Importing the libraries

import time
import math
import seaborn
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import sqrt
from numpy import array
from keras import regularizers
from sklearn import preprocessing
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

# Data Cleaning
dataset_mentah = pd.read_csv('data/B5_1-10_3dtk.csv', index_col=False, parse_dates=['waktu'])

dataset3 = dataset_mentah[['waktu','suhu_2', 'suhu_permukaan','kelembaban_tanah']]
dataset3.set_index('waktu', inplace=True)

# #Split data 1 Hari
dataset3 = dataset3['2020-05-07 12:30:00':'2020-05-07 12:35:00']
print(dataset3)

#Freq jadi 1 Detik
detik= dataset3.resample('3S').pad().dropna()

#Isi data di detik yang kosong
detik=detik.ffill(axis = 0)
print(detik.head(20))

#Simpan csv
# detik.to_csv("B5_1-_3dtk.csv")

# Tampilkan Diagram
dataset3['kelembaban_tanah'].plot()
plt.show()