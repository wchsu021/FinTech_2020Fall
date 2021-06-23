# -*- coding: utf-8 -*-
"""fintech_hw4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P-j5fX9iqPs0XG0opeUuhxirZwOWh7qU
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, GRU, Activation, Dense, Dropout, Input
from keras.optimizers import Adam

from sklearn.metrics import mean_squared_error

def data_preprocess(data):
    cols = list(data)
    cols.pop(0)
    for col in cols:
        # print(col)
        data[col]=(data[col]-data[col].min())/(data[col].max()-data[col].min())
    data.drop(columns=["Adj Close"],inplace=True)
    return data

chart = pd.read_csv('feature.csv')
input_data = data_preprocess(chart)

def gen_input_data(data, time_step = 30, future_day = 1):

    data_date = data["Date"]
    # print(data_date)
    data = data.drop(["Date"], axis = 1)
    date_list = []
    train_x, train_y = [], []
    for i in range(data.shape[0]-time_step):
        # print(data_date[i])
        date_list.append(data_date[i+time_step])
        train_x.append(np.array(data.iloc[i:i+time_step]))
        train_y.append(np.array(data.iloc[i+time_step:i+time_step+future_day]["Close"]))
    return np.array(train_x), np.array(train_y), date_list

input_data.tail()

train_x, train_y, date_list = gen_input_data(input_data)

print(train_x.shape, train_y.shape, len(date_list))

print(date_list[-1])

print(date_list[0])

target_date = 0
for idx, d in enumerate(date_list):
    if d > "2019-01-01":
        print(idx, d)
        target_date = idx
        break
print(target_date)

# valid_x = train_x[6235:]
# valid_y = train_y[6235:]
# train_x = train_x[:6235]
# train_y = train_y[:6235]

valid_x = train_x[target_date:]
valid_y = train_y[target_date:]
train_x = train_x[:target_date]
train_y = train_y[:target_date]

def LSTM_model(shape):

	model = Sequential()
	model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
	# output shape: (1, 1)
	model.add(Dense(1))
	model.compile(loss="mse", optimizer="adam")
	model.summary()
	return model

model_LSTM = LSTM_model(train_x.shape)

print(model_LSTM)

history_LSTM = model_LSTM.fit(train_x, train_y, epochs = 30, batch_size = 32, validation_data=(valid_x, valid_y))

loss_LSTM = history_LSTM.history['loss']
val_loss_LSTM = history_LSTM.history['val_loss']
epochs_LSTM = range(1, len(loss_LSTM) + 1)

# plt.figure()
plt.plot(epochs_LSTM, loss_LSTM, label='Training loss')
plt.plot(epochs_LSTM, val_loss_LSTM, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predict_LSTM = model_LSTM.predict(valid_x)

# print(train_y)

# print(predict_y)

# print(valid_y)

py_LSTM = predict_LSTM.reshape((-1,))
# py

vy = valid_y.reshape((-1,))
# vy

mse_LSTM = mean_squared_error(vy, py_LSTM)
print(mse_LSTM)

plt.plot(vy, color = 'red', label = 'Real Stock Price')  # 紅線表示真實股價
plt.plot(py_LSTM, color = 'blue', label = 'Predicted Stock Price')  # 藍線表示預測股價
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('tock Price')
plt.legend()
plt.show()

def SimpleRNN_model(shape):

	batch_input_shape = (None, shape[1], shape[2])
	print(batch_input_shape)
	model = Sequential()
	model.add(SimpleRNN(64, batch_input_shape = batch_input_shape, unroll=True))
	# model.add(Dropout(0.2))
	model.add(Dense(1))
	# model.add(Activation('softmax'))
	model.compile(loss = "mse", optimizer = "adam")
	# model.summary()

	return model

model_RNN = SimpleRNN_model(train_x.shape)

print(model_RNN)

history_RNN = model_RNN.fit(train_x, train_y, epochs = 30, batch_size = 32, validation_data=(valid_x, valid_y))

loss_RNN = history_RNN.history['loss']
val_loss_RNN = history_RNN.history['val_loss']
epochs_RNN = range(1, len(loss_RNN) + 1)

# plt.figure()
plt.plot(epochs_RNN, loss_RNN, label='Training loss')
plt.plot(epochs_RNN, val_loss_RNN, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predict_RNN = model_RNN.predict(valid_x)
py_RNN = predict_RNN.reshape((-1,))
vy = valid_y.reshape((-1,))

mse_RNN = mean_squared_error(vy, py_RNN)
print(mse_RNN)

plt.plot(vy, color = 'red', label = 'Real Stock Price')  # 紅線表示真實股價
plt.plot(py_RNN, color = 'blue', label = 'Predicted Stock Price')  # 藍線表示預測股價
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('tock Price')
plt.legend()
plt.show()

def GRU_model(shape):

	model = Sequential()
	model.add(GRU(10, input_length=shape[1], input_dim=shape[2]))
	# output shape: (1, 1)
	model.add(Dense(1))
	model.compile(loss="mse", optimizer="adam")
	model.summary()
	return model

model_GRU = GRU_model(train_x.shape)

print(model_GRU)

history_GRU = model_GRU.fit(train_x, train_y, epochs = 30, batch_size = 32, validation_data=(valid_x, valid_y))

loss_GRU = history_GRU.history['loss']
val_loss_GRU = history_GRU.history['val_loss']
epochs_GRU = range(1, len(loss_GRU) + 1)

# plt.figure()
plt.plot(epochs_GRU, loss_GRU, label='Training loss')
plt.plot(epochs_GRU, val_loss_GRU, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predict_GRU = model_GRU.predict(valid_x)
py_GRU = predict_GRU.reshape((-1,))
vy = valid_y.reshape((-1,))

mse_GRU = mean_squared_error(vy, py_GRU)
print(mse_GRU)

plt.plot(vy, color = 'red', label = 'Real Stock Price')  # 紅線表示真實股價
plt.plot(py_GRU, color = 'blue', label = 'Predicted Stock Price')  # 藍線表示預測股價
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('tock Price')
plt.legend()
plt.show()

plt.plot(vy, color = 'red', label = 'Real Stock Price')  # 紅線表示真實股價
plt.plot(py_GRU, color = 'blue', label = 'GRU')  # 藍線表示預測股價
plt.plot(py_RNN, color = 'green', label = 'RNN')  # 藍線表示預測股價
plt.plot(py_LSTM, color = 'orange', label = 'LSTM')  # 藍線表示預測股價
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('tock Price')
plt.legend()
plt.show()

chart_2020 = pd.read_csv('feature_2020.csv')
input_data_2020 = data_preprocess(chart_2020)

test_x, test_y, date_list_2020 = gen_input_data(input_data_2020)

target_data = 0
for idx, d in enumerate(date_list_2020):
    if d > "2020-01-01":
        print(idx, d)
        target_data = idx
        break

test_x = test_x[target_data:]
test_y = test_y[target_data:]

predict_GRU_2020 = model_GRU.predict(test_x)
predict_RNN_2020 = model_RNN.predict(test_x)
predict_LSTM_2020 = model_LSTM.predict(test_x)
py_GRU_2020 = predict_GRU_2020.reshape((-1,))
py_RNN_2020 = predict_RNN_2020.reshape((-1,))
py_LSTM_2020 = predict_LSTM_2020.reshape((-1,))
vy = test_y.reshape((-1,))

mse_RNN_2020 = mean_squared_error(vy, py_RNN_2020)
mse_LSTM_2020 = mean_squared_error(vy, py_LSTM_2020)
mse_GRU_2020 = mean_squared_error(vy, py_GRU_2020)
print(mse_GRU_2020, mse_LSTM_2020, mse_RNN_2020)

plt.plot(vy, color = 'red', label = 'Real Stock Price')  # 紅線表示真實股價
plt.plot(py_GRU_2020, color = 'blue', label = 'GRU')  # 藍線表示預測股價
plt.plot(py_RNN_2020, color = 'green', label = 'RNN')  # 藍線表示預測股價
plt.plot(py_LSTM_2020, color = 'orange', label = 'LSTM')  # 藍線表示預測股價
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('tock Price')
plt.legend()
plt.show()