import keras.regularizers
import tushare as ts
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import SimpleRNN, Dropout, Dense, LSTM
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def get_stock_data():
    sc = input('Please Enter Stock code: ')
    df = input('Date from xxxx-xx-xx: ')
    dt = input('Date from xxxx-xx-xx: ')
    data = ts.get_k_data(sc, ktype='D', start=df, end=dt)
    data_path = "./data/{}.csv".format(sc)
    data.to_csv(data_path)


dp = "./data/{}.csv".format(input('Please Enter Stock code: '))
rows = len(open(dp).readlines())    # 获取总行数
stock_data = pd.read_csv(dp)

train_data = stock_data.iloc[0:rows - 600, 2:3].values
test_data = stock_data.iloc[rows-600:, 2:3].values

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(120, len(train_data)):
    x_train.append(train_data[i-120:i, 0])
    y_train.append(train_data[i, 0])

for i in range(120, len(test_data)):
    x_test.append(test_data[i-120:i, 0])
    y_test.append(test_data[i, 0])

np.random.seed(666)
np.random.shuffle(x_train)
np.random.seed(666)
np.random.shuffle(y_train)
tf.random.set_seed(666)

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], 120, 1))

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 120, 1))

model = tf.keras.models.Sequential([
    LSTM(120, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.001)),
    Dropout(0.4),
    LSTM(140, kernel_regularizer=keras.regularizers.l2(0.001)),
    Dropout(0.4),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

checkpoint_save_path = './Checkpoint/Stock_forecast.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test), callbacks=[cp_callback])

model.summary()

predict_result = model.predict(x_test)
print(predict_result.shape)
for i in range(120):
    next_input = [predict_result[-120:, 0]]
    next_input = np.array(next_input)
    next_input = np.reshape(next_input, (1, 120, 1))
    next_predict = model.predict(next_input)
    predict_result = np.concatenate((predict_result, next_predict), axis=0)


# 还原数据:
predict_result = scaler.inverse_transform(predict_result)
real_result = scaler.inverse_transform(test_data[120:])

# 绘制预测结果
plt.plot(real_result, color='green', label='RealPrice')
plt.plot(predict_result, color='red', label='PredictPrice')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('RNN stock prediction')
plt.legend()


plt.show()
