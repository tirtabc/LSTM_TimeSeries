from flask import Flask, render_template, jsonify
from sqlalchemy import create_engine
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pickle
import scipy.stats as stats
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)

connection_url = 'mysql+pymysql://root:@localhost/Data_Bulan_3-5'
connection = create_engine(connection_url)

KELEMBABAN = 'kelembaban_tanah'
SUHU_PERMUKAAN = 'suhu_permukaan'
SUHU_TANAH = 'suhu_tanah'

time_steps = 10
lstm = pickle.load(open('model/lstm_B5_1-7_3dtk.pkl', 'rb'))
Kor1 = pickle.load(open('model/lstm_Korelasi_kel-suhu.pkl', 'rb'))
Kor2 = pickle.load(open('model/lstm_Korelasi_kel-permukaan.pkl', 'rb'))
Interval = pickle.load(open('model/lstm_interval.pkl', 'rb'))
print('kor1', Kor1)
print('kor2', Kor2)
print('interval', Interval)
# scaler = pickle.load(open('model/scaler_3dtk.pkl', 'rb'))
# scaler2 = pickle.load(open('model/scaler2_3dtk.pkl', 'rb'))

kelembaban = pd.read_sql("SELECT CONCAT(tanggal, ' ', jam) AS waktu, "
                         "kelembaban_tanah "
                         "FROM data_kel_tanah "
                         "WHERE id_alat = 'K6' AND tanggal LIKE '2020-05-11' AND jam BETWEEN '12:29:00' AND '12:33:00' ",
                         con=connection)
kelembaban['waktu'] = pd.to_datetime(kelembaban['waktu'])
kelembaban = kelembaban.set_index('waktu').sort_values('waktu')
kelembaban = kelembaban.resample('3S').pad().dropna()
print(kelembaban)

suhu_tanah = pd.read_sql("SELECT CONCAT(tanggal, ' ', jam) AS waktu, suhu_tanah "
                         "FROM data_su_tanah "
                         "WHERE id_alat = 'S4' AND tanggal LIKE '2020-05-11' AND jam BETWEEN '12:29:00' AND '12:33:00'",
                         con=connection)
suhu_tanah['waktu'] = pd.to_datetime(suhu_tanah['waktu'])
suhu_tanah = suhu_tanah.set_index('waktu').sort_values('waktu')
suhu_tanah = suhu_tanah.resample('3S').pad().dropna()
print(suhu_tanah)

suhu_permukaan = pd.read_sql("SELECT CONCAT(tanggal, ' ', jam) AS waktu, suhu_tanah AS suhu_permukaan "
                             "FROM data_su_tanah "
                             "WHERE id_alat = 'S5' AND tanggal LIKE '2020-05-11' AND jam BETWEEN '12:29:00' AND '12:33:00' ",
                             con=connection)
suhu_permukaan['waktu'] = pd.to_datetime(suhu_permukaan['waktu'])
suhu_permukaan = suhu_permukaan.set_index('waktu').sort_values('waktu')
suhu_permukaan = suhu_permukaan.resample('3S').pad().dropna()

data = pd.merge_asof(left=kelembaban, right=suhu_tanah, left_index=True, right_index=True)
data = pd.merge_asof(left=data, right=suhu_permukaan, left_index=True, right_index=True)

index = 0


@app.route("/")
def main():
    Korelasi1 =Kor1[0,1]
    Korelasi2 =Kor2[0,1]
    return render_template('index.html', kor1 = Korelasi1, kor2=Korelasi2)


@app.route("/ajax", methods=['GET'])
def get_recent_data():
    global index
    # Data Bahan
    data1 = data[index:time_steps + index]
    print('data bahan ', data1)
    # Data Asli
    data2 = data.iloc[index + time_steps].squeeze()
    print('data asli ', data2)

    scaler = MinMaxScaler()
    scaler2 = MinMaxScaler()
    scaler = scaler.fit(data1[[SUHU_TANAH, SUHU_PERMUKAAN]].to_numpy())
    scaler2 = scaler2.fit(data1[[KELEMBABAN]])
    print('data_scales2', data)
    print('data_scales2 shape', data)

    # Transform Data Bahan jadi X_test
    data1.loc[:, [SUHU_TANAH, SUHU_PERMUKAAN]] = scaler.transform(data1[[SUHU_TANAH, SUHU_PERMUKAAN]].to_numpy())
    data1[KELEMBABAN] = scaler2.transform(data1[[KELEMBABAN]])
    print('data1', data1)
    X_test = np.array([data1])

    # Prediksi Data Bahan
    y_pred = lstm.predict(X_test)
    print('y_pred sebelum inverse', y_pred)
    y_pred = scaler2.inverse_transform(y_pred)[0][0]
    print('y_pred setelah invers', y_pred)
    # Hasil Sebenarnya Data Bahan
    y_test = data2[KELEMBABAN]

    print(y_pred, y_test)

    # y_test_inv = kelembaban_tanah_transformer.inverse_transform(y_test.reshape(1, -1))
    # y_pred_inv = kelembaban_tanah_transformer.inverse_transform(y_pred)
    #
    # print('y_test_inv', y_test_inv, 'len y_test_inv ', len(y_test_inv.ravel()))
    # print('y_pred_inv', y_pred_inv, 'len y_pred_inv ', len(y_pred_inv.ravel()))
    # # estimate stdev of yhat
    # sum_errs = np.sum((y_test_inv.ravel() - y_pred_inv.ravel()) ** 2)
    # print('sum_errs', sum_errs)
    # stdev = np.sqrt(1 / (len(y_test_inv.ravel()) - 2) * sum_errs)
    # print('stdev', stdev)
    # # calculate prediction interval
    # interval = 1.96 * stdev
    # print('Prediction Interval: %.3f' % interval)

    # (kelembaban  atnah dan suhu_tanah)
    koefisien1 = 1, 0.2047054
    # (kelembaban  atnah dan suhu_permukaan)
    koefisien2 = 1, - 0.51365219
    # 8.06 |  0.602

    prediction_interval = Interval
    batas_atas, batas_bawah = y_pred + prediction_interval, y_pred - prediction_interval
    print('pa sebelum', batas_atas)
    batas_atas = math.ceil(batas_atas)
    print('pa sesudah', batas_atas)

    print('pb sebelum', batas_bawah)
    batas_bawah = math.floor(batas_bawah)
    print('p sesudah', batas_bawah)
    is_outlier = bool(y_test < batas_bawah or y_test > batas_atas)

    # Untuk simpan data
    pd.DataFrame({
        'waktu': [data2.name],
        'kelembaban_tanah': [y_test],
        'prediksi_kelembaban_tanah': [y_pred],
        'min_kelembaban': [batas_bawah],
        'max_kelembaban': [batas_atas],
        'suhu_tanah': data2['suhu_tanah'],
        'suhu_permukaan': data2['suhu_permukaan'],
        'is_outlier_kelembaban_tanah': [is_outlier],
    }).to_sql('filter_data', connection, if_exists='append', index=False)

    index += 1
    formatted_df = data2.name.strftime("%m/%d/%Y, %H:%M:%S")
    return jsonify({
        'index2': index,
        'waktu': formatted_df,
        'kelembaban': {
            'asli': float(y_test),
            'prediksi': float(y_pred),
            'anomali': bool(is_outlier),
            'upper': float(batas_atas),
            'lower': float(batas_bawah)
        },
        'suhu_tanah': int(data2['suhu_tanah']),
        'suhu_permukaan': int(data2['suhu_permukaan']),
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
