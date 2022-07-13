from operator import methodcaller

from flask import Flask, render_template, session, jsonify, request
import random
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import pickle
from datetime import datetime
import numpy as np
import math
import time
import scipy.stats as stats
from sqlalchemy import create_engine
import pymysql

# TODO : 1. bikin model yang bagus dana da dasarnya, sering di coba unutk anomali apa gitu, buat contoh, prediksi

connection_url = 'mysql+pymysql://root:@localhost/Data_Bulan_3-5'
connection = create_engine(connection_url)

kelembaban = pd.read_sql("SELECT CONCAT(tanggal, ' ', jam) AS waktu, kelembaban_tanah "
                         "FROM data_kel_tanah "
                         "WHERE id_alat = 'K5' and tanggal like '2020-03-09' "
                         "ORDER BY nomor DESC "
                         "LIMIT 60",
                         con=connection)
kelembaban['waktu'] = pd.to_datetime(kelembaban['waktu'])
kelembaban = kelembaban.set_index('waktu').sort_values('waktu')
# cursor.execute(
#     "SELECT CONCAT(tanggal, ' ', jam) AS waktu, suhu_tanah FROM data_su_tanah WHERE id_alat = 'S5' and tanggal like '2020-03-09' ORDER BY nomor DESC LIMIT 60")
# suhu_tanah = cursor.fetchall()
# cursor.execute(
#     "SELECT CONCAT(tanggal, ' ', jam) AS waktu, suhu_tanah FROM data_su_tanah WHERE id_alat = 'S4' and tanggal like '2020-03-09' ORDER BY nomor DESC LIMIT 60")
# suhu_permukaan = cursor.fetchall()
curah_hujan = pd.read_sql("SELECT CONCAT(tanggal, ' ', jam) AS waktu, curah_hujan "
                          "FROM data_hujan "
                          "WHERE id_alat = 'ARR003' and tanggal like '2020-03-09' "
                          "ORDER BY nomor DESC "
                          "LIMIT 61",
                          con=connection)
curah_hujan['waktu'] = pd.to_datetime(curah_hujan['waktu'])
curah_hujan = curah_hujan.set_index('waktu').sort_values('waktu').diff(periods=-1).dropna()


print(kelembaban.head())
print(curah_hujan.head())

hasil = pd.merge_asof(left=curah_hujan, right=kelembaban, left_index=True, right_index=True,
    allow_exact_matches=False)

print(hasil)