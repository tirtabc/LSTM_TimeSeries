from flask import Flask, render_template, jsonify
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
import pandas as pd
import pickle
import scipy.stats as stats

app2 = Flask(__name__)

connection_url = 'mysql+pymysql://root:@localhost/Data_Bulan_3-5'
connection = create_engine(connection_url)


# Todo ::   1. Buat tampilan perhari. 2. Buat multivariate untuk LSTM.
#           3. Buat corellation 4. Prediksi hujan di hapus
#           5. Jadiin 1 folder dari ambil data csv, olahnya, buat modelnya, prediksinya, sampe flasknya


@app2.route("/")
def main():
    return render_template('index2.html')


cache = False
lstm = None
scaler = None


@app2.route("/ajax", methods=['GET'])
def get_recent_data():
    global lstm
    # global scaler
    # global scaler2
    global cache
    if cache is False:
        lstm = pickle.load(open('model/lstm_3dtk.pkl', 'rb'))
        # scaler = pickle.load(open('model/scaler3.pkl', 'rb'))
        # scaler2 = pickle.load(open('model/scaler3_2.pkl', 'rb'))
        cache = True

    kelembaban = pd.read_sql("SELECT CONCAT(tanggal, ' ', jam) AS waktu, kelembaban_tanah "
                             "FROM data_kel_tanah "
                             "WHERE id_alat = 'K6' and tanggal like '2020-04-11' and jam BETWEEN '00:00:01' AND '00:03:00'  "
                             "ORDER BY nomor DESC "
                             "LIMIT 31",
                             con=connection)
    kelembaban['waktu'] = pd.to_datetime(kelembaban['waktu'])
    kelembaban = kelembaban.set_index('waktu').sort_values('waktu')

    suhu_tanah = pd.read_sql("SELECT CONCAT(tanggal, ' ', jam) AS waktu, suhu_tanah "
                             "FROM data_su_tanah "
                             "WHERE id_alat = 'S4' and tanggal like '2020-04-16' and jam BETWEEN '00:00:01' AND '00:03:00'   "
                             "ORDER BY nomor DESC "
                             "LIMIT 31",
                             con=connection)
    suhu_tanah['waktu'] = pd.to_datetime(suhu_tanah['waktu'])
    suhu_tanah = suhu_tanah.set_index('waktu').sort_values('waktu')

    suhu_permukaan = pd.read_sql("SELECT CONCAT(tanggal, ' ', jam) AS waktu, suhu_tanah AS suhu_permukaan "
                                 "FROM data_su_tanah "
                                 "WHERE id_alat = 'S5' and tanggal like '2020-04-16' and jam BETWEEN '00:00:01' AND '00:03:00'   "
                                 "ORDER BY nomor DESC "
                                 "LIMIT 31",
                                 con=connection)
    suhu_permukaan['waktu'] = pd.to_datetime(suhu_permukaan['waktu'])
    suhu_permukaan = suhu_permukaan.set_index('waktu').sort_values('waktu')

    # print(kelembaban,suhu_tanah)
    data = pd.merge_asof(left=kelembaban, right=suhu_tanah, left_index=True, right_index=True)
    # print("sini 1")
    # print(kelembaban, suhu_permukaan)
    data = pd.merge_asof(left=data, right=suhu_permukaan, left_index=True, right_index=True)
    print('data', data)
    data_asli = data.iloc[-1]
    data_bahan = data.drop(data_asli.name)

    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    f_columns = ['suhu_tanah', 'suhu_permukaan']
    data1 = data.loc[:, f_columns]
    print('f_columns', data1)
    print('f_columns shape', data1.shape)

    scaler = MinMaxScaler()
    scaler2 = MinMaxScaler()

    scaler = scaler.fit(data[f_columns].to_numpy())
    scaler2 = scaler2.fit(data[['kelembaban_tanah']])
    print('data_scales2', data)
    print('data_scales2 shape', data)

    data.loc[:, f_columns] = scaler.transform(data[f_columns].to_numpy())
    data['kelembaban_tanah'] = scaler2.transform(data[['kelembaban_tanah']])
    print('data_scales', data)
    print('data_scales shape', data)

    time_steps = 10
    X_test, y_test = create_dataset(data, data.kelembaban_tanah, time_steps)
    print('X_pred', X_test)
    print('X_pred shape', X_test.shape)
    print('Y_pred', y_test)
    print('Y_pred shpae', y_test.shape)

    # X_pred_scaled = scaler.transform(X_pred)
    # print(X_pred_scaled)
    y_pred = lstm.predict(X_test)
    # y_test = scaler2.inverse_transform(y_test.reshape(1, -1))
    y_pred = scaler2.inverse_transform(y_pred)
    # y_pred = y_pred[0]
    # y_pred = np.round(y_pred).astype(int)
    print("prediksi:", y_pred)
    print("prediksi shape:", y_pred.shape)

    # X_pred = data_bahan.to_numpy()
    # X_pred_scaled = scaler.transform(X_pred)
    # X_pred_scaled = np.reshape(X_pred_scaled, (1, 60, 3))
    # y_pred = lstm.predict(X_pred_scaled)
    # y_pred = scaler.inverse_transform(y_pred)
    # # Cuma mau ambil prediksi pertama
    # y_pred = y_pred[0]
    # print("prediksi:", y_pred)
    bound_bahan = data_bahan.to_numpy()
    bound_2 = np.array(bound_bahan[:, 0])
    print('bound_2', bound_2, 'bound_2 type', type(bound_2))

    H_pred = y_pred[-3:].ravel()
    print('y_pred', H_pred, 'y_pred', type(H_pred))
    bound_1 = np.concatenate([bound_2, H_pred])
    print('bound_1 + ypred :', bound_1)

    def threshold(x_pred1):
        xbar_sample = x_pred1.mean()
        # print("Xbar_sample = \n", xbar_sample)
        sigma_sample = x_pred1.std()
        # print("Sigma sample = \n", sigma_sample)
        SE_sample = sigma_sample / math.sqrt(data_bahan.size)
        # print("SE_sample = \n", SE_sample)
        # z score = mean+- 1.96 σ/akar dari n
        z_critical = stats.norm.ppf(q=0.95)
        # print("Z_Critical = \n", z_critical)
        # bb = xbar_sample - z_critical * SE_sample
        bb = xbar_sample - z_critical * SE_sample
        bb= math.floor(bb)
        # ba = xbar_sample + z_critical * SE_sample
        ba = xbar_sample + z_critical * SE_sample
        ba = math.ceil(ba)
        return ba, bb

    bound_kel = list(threshold(bound_1[-5:]))
    print('Bound atas, bound bawah', bound_kel, bound_kel[0], bound_kel[1])

    bound = pd.DataFrame({
        'kelembaban_tanah': {
            'upper': bound_kel[0],
            'lower': bound_kel[1],
        },
    })

    prediksi_kelembaban_tanah = float(y_pred[0])
    is_outlier_kelembaban_tanah = bool(data_asli['kelembaban_tanah'] < bound['kelembaban_tanah']['lower'] or
                                       data_asli['kelembaban_tanah'] > bound['kelembaban_tanah']['upper'])

    waktu = data_asli.name

    response = pd.DataFrame({
        'waktu': [waktu],
        'kelembaban_tanah': [data_asli['kelembaban_tanah']],
        'is_outlier_kelembaban_tanah': [is_outlier_kelembaban_tanah],
        'prediksi_kelembaban_tanah': [prediksi_kelembaban_tanah],
    })

    filter1 = pd.read_sql("SELECT * "
                          "FROM filter1 "
                          "ORDER BY id DESC "
                          "LIMIT 1",
                          con=connection)
    if len(filter1) == 0:
        response.to_sql('filter1', connection, if_exists='append', index=False)
        print("kosong")
    else:
        filter1 = filter1.iloc[0]
        if filter1['kelembaban_tanah'] != data_asli['kelembaban_tanah'] :
            response.to_sql('filter1', connection, if_exists='append', index=False)
            print('tidak sama')
        else:
            print('sama')

    print(bound['kelembaban_tanah']['upper'])

    return jsonify({
        'waktu': data_asli.name,
        'kelembaban': {
            'asli': float(data_asli['kelembaban_tanah']),
            'prediksi': float(y_pred[0]),
            'anomali': bool(data_asli['kelembaban_tanah'] < bound['kelembaban_tanah']['lower'] or
                            data_asli['kelembaban_tanah'] > bound['kelembaban_tanah']['upper']),
            'upper': float(bound['kelembaban_tanah']['upper']),
            'lower': float(bound['kelembaban_tanah']['lower'])
        },
    })


if __name__ == "__main__":
    app2.run(debug=True, host='0.0.0.0', port=8000)
