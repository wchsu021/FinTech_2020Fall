# -*- coding: utf-8 -*-
"""Bidirectional_Taiwan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m0QzGeqM9Q35PH-3xqwVPKJPwyJrA7oZ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, GRU, Activation, Dense, Dropout, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

"""#0.Data Preprocess"""

stock_pd = pd.read_csv('Taiwan_stock_short_all.csv')
covid_pd = pd.read_csv('Taiwan_covid.csv')

stock_pd

def stock_preprocess(stock_pd):

    stock_pd = stock_pd.dropna()
    # stock_pd.drop(columns=["Adj Close", "Volume"],inplace=True)
    stock_pd.drop(columns=["Adj Close"],inplace=True)

    for i in range(len(stock_pd['Date'])):
        temp = stock_pd.loc[i,'Date']
        if '/' in temp:
            # print(covid_pd.loc[i,'Date'])
            
            temp = time.strptime(temp, "%Y/%m/%d")
            nt = time.strftime("%Y-%m-%d", temp)
            # print(nt)
            stock_pd.loc[i,'Date'] = nt

    return stock_pd

def covid_preprocess(covid_pd):

    covid_pd = covid_pd.rename(columns = {'date': 'Date'}, inplace = False)

    features = list(covid_pd.columns)
    print(features)
    features = features[:2]
    covid_pd = covid_pd[features]

    covid_pd = covid_pd.dropna()

    for i in range(len(covid_pd['Date'])):
        temp = covid_pd.loc[i,'Date']
        if '/' in temp:
            # print(covid_pd.loc[i,'Date'])
            
            temp = time.strptime(temp, "%Y/%m/%d")
            nt = time.strftime("%Y-%m-%d", temp)
            # print(nt)
            covid_pd.loc[i,'Date'] = nt

    return covid_pd

stock_pd = stock_preprocess(stock_pd)
covid_pd = covid_preprocess(covid_pd)

stock_pd

# stock_pd = stock_pd[(stock_pd['Date'] >= "2020-02-01") & (stock_pd['Date'] <= "2020-11-30")]
# covid_pd = covid_pd[(covid_pd['Date'] >= "2020-02-01") & (covid_pd['Date'] <= "2020-11-30")]
stock_pd = stock_pd[(stock_pd['Date'] <= "2020-11-30")]
covid_pd = covid_pd[(covid_pd['Date'] <= "2020-11-30")]

stock_pd = stock_pd.reset_index(drop=True)
covid_pd = covid_pd.reset_index(drop=True)

stock_pd

covid_pd

def merge_pd(stock_pd, covid_pd):

    case = 0
    case_list = []
    for idx_c, date_c in enumerate(list(covid_pd['Date'])):
        case = case + covid_pd.loc[idx_c, "confirmed_cases"]
        if date_c in list(stock_pd['Date']):
            # print(idx_c, date_c)
            case_list.append(case)
            case = 0
        else:
            # case = case + covid_pd.loc[idx_c, "confirmed_cases"]
            pass
    
    print(len(case_list))
    print(len(stock_pd))
    ls = len(stock_pd)
    lc = len(case_list)
    for i in range(ls-lc):
        case_list.insert(0, 0)
    stock_pd["covid"]=case_list
    return stock_pd

df = merge_pd(stock_pd, covid_pd)

df.to_csv("Tw_data.csv", index = False)

# df = stock_pd.copy()

def normalize(df, cols):
    train_set_normalized = df.copy()
    for col in cols:
        all_col_data = train_set_normalized[col].copy()
        # print(all_col_data)
        mu = all_col_data.mean()
        std = all_col_data.std()
        
        z_score_normalized = (all_col_data - mu) / std
        train_set_normalized[col] = z_score_normalized
    return train_set_normalized

def normalize_v2(df, cols):
    '''
    train_set_normalized = df.copy()
    x = train_set_normalized.values #returns a numpy array
    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    train_set_normalized = pd.DataFrame(x_scaled)
    return train_set_normalized
    '''
    train_set_normalized = df.copy()
    scaler = preprocessing.StandardScaler()
    for col in cols:
        train_set_normalized[col] = scaler.fit_transform(train_set_normalized[col].values.reshape(-1,1))
    return train_set_normalized

def normalize_v3(df, cols):
    '''
    train_set_normalized = df.copy()
    x = train_set_normalized.values #returns a numpy array
    scaler = preprocessing.StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    train_set_normalized = pd.DataFrame(x_scaled)
    return train_set_normalized
    '''
    train_set_normalized = df.copy()
    scaler = preprocessing.MinMaxScaler()
    for col in cols:
        train_set_normalized[col] = scaler.fit_transform(train_set_normalized[col].values.reshape(-1,1))
    return train_set_normalized

# scaler_c = preprocessing.StandardScaler()
# scaler_c.fit(df['Close'].values.reshape(-1,1))

scaler_c = preprocessing.MinMaxScaler()
scaler_c.fit(df['Close'].values.reshape(-1,1))

cols = list(df)
cols.pop(0)
# df = normalize_v2(df, cols)
df = normalize_v3(df, cols)

"""#1. Generate training dataset"""

def gen_input_data(data, time_step = 5, future_day = 1):

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

train_x, train_y, date_list = gen_input_data(df)

print(train_x.shape, train_y.shape, len(date_list))

target_date = 0
for idx, d in enumerate(date_list):
    if d > "2020-09-30":
        print(idx, d)
        target_date = idx
        break
print(target_date)

valid_x = train_x[target_date:]
valid_y = train_y[target_date:]
train_x = train_x[:target_date]
train_y = train_y[:target_date]

print(valid_y.shape, train_y.shape)

def shuffle(X,Y):
  np.random.seed(10)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]

train_x, train_y = shuffle(train_x, train_y)

"""#2. Model"""

def LSTM_model(shape):

	model = Sequential()
	model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
	# output shape: (1, 1)
	model.add(Dense(1))
	model.compile(loss="mse", optimizer="adam")
	model.summary()
	return model

def GRU_model(shape):

	model = Sequential()
	model.add(GRU(10, input_length=shape[1], input_dim=shape[2]))
	# output shape: (1, 1)
	model.add(Dense(1))
	model.compile(loss="mse", optimizer="adam")
	model.summary()
	return model

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

def BDLSTM_model(shape):
    
    model = Sequential()

    # forward_l = LSTM(64, return_sequences=False)
    # backward_l = LSTM(64, activation='relu', return_sequences=False, go_backwards=True)
    # # model.add(Bidirectional(forward_l, backward_layer=backward_l, input_shape=(shape[1], shape[2])))
    # model.add(Bidirectional(forward_l, backward_layer=backward_l))
    # # layer, merge_mode="concat", weights=None, backward_layer=None

    model.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(shape[1], shape[2])))
    model.add(Bidirectional(LSTM(64)))

    model.add(Dense(1))
    # model.add(Activation('softmax'))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model
    
    '''
    model.add(LSTM(64, input_shape=(shape[1], shape[2]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(64, input_shape=(shape[1], shape[2]), return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))
    # model = load_model('my_LSTM_stock_model1000.h5')
    adam = keras.optimizers.Adam(decay=0.2)
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
    '''

# model_LSTM = LSTM_model(train_x.shape)

# model_GRU = GRU_model(train_x.shape)

# model_RNN = SimpleRNN_model(train_x.shape)

model_BDLSTM = BDLSTM_model(train_x.shape)

"""#3. Train"""

filepath="weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

"""LSTM"""

# history_LSTM = model_LSTM.fit(train_x, train_y, epochs = 30, batch_size = 32, validation_data=(valid_x, valid_y))

# loss_LSTM = history_LSTM.history['loss']
# val_loss_LSTM = history_LSTM.history['val_loss']
# epochs_LSTM = range(1, len(loss_LSTM) + 1)

# # plt.figure()
# plt.plot(epochs_LSTM, loss_LSTM, label='Training loss')
# plt.plot(epochs_LSTM, val_loss_LSTM, label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

"""GRU"""

# history_GRU = model_GRU.fit(train_x, train_y, epochs = 30, batch_size = 32, validation_data=(valid_x, valid_y))

# loss_GRU = history_GRU.history['loss']
# val_loss_GRU = history_GRU.history['val_loss']
# epochs_GRU = range(1, len(loss_GRU) + 1)

# # plt.figure()
# plt.plot(epochs_GRU, loss_GRU, label='Training loss')
# plt.plot(epochs_GRU, val_loss_GRU, label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

"""SimpleRNN"""

# history_RNN = model_RNN.fit(train_x, train_y, epochs = 30, batch_size = 32, validation_data=(valid_x, valid_y))

# loss_RNN = history_RNN.history['loss']
# val_loss_RNN = history_RNN.history['val_loss']
# epochs_RNN = range(1, len(loss_RNN) + 1)

# # plt.figure()
# plt.plot(epochs_RNN, loss_RNN, label='Training loss')
# plt.plot(epochs_RNN, val_loss_RNN, label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

"""BDLSTM"""

# history_BDLSTM = model_BDLSTM.fit(train_x, train_y, epochs = 30, batch_size = 16, validation_data=(valid_x, valid_y), callbacks=callbacks_list)
history_BDLSTM = model_BDLSTM.fit(train_x, train_y, epochs = 30, batch_size = 8, validation_split=0.2, callbacks=callbacks_list)

loss_BDLSTM = history_BDLSTM.history['loss']
val_loss_BDLSTM = history_BDLSTM.history['val_loss']
epochs_BDLSTM = range(1, len(loss_BDLSTM) + 1)

# plt.figure()
plt.plot(epochs_BDLSTM, loss_BDLSTM, label='Training loss')
plt.plot(epochs_BDLSTM, val_loss_BDLSTM, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

"""#4. Predict"""

vy = valid_y.reshape((-1,))

"""LSTM"""

# predict_LSTM = model_LSTM.predict(valid_x)
# py_LSTM = predict_LSTM.reshape((-1,))

# mse_LSTM = mean_squared_error(vy, py_LSTM)
# print(mse_LSTM)

# plt.plot(vy, color = 'red', label = 'Real Stock Price')  # 紅線表示真實股價
# plt.plot(py_LSTM, color = 'blue', label = 'Predicted Stock Price')  # 藍線表示預測股價
# plt.title('Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()

"""GRU"""

# predict_GRU = model_GRU.predict(valid_x)
# py_GRU = predict_GRU.reshape((-1,))
# # vy = valid_y.reshape((-1,))

# mse_GRU = mean_squared_error(vy, py_GRU)
# print(mse_GRU)

# plt.plot(vy, color = 'red', label = 'Real Stock Price')  # 紅線表示真實股價
# plt.plot(py_GRU, color = 'blue', label = 'Predicted Stock Price')  # 藍線表示預測股價
# plt.title('Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()

"""SimpleRNN"""

# predict_RNN = model_RNN.predict(valid_x)
# py_RNN = predict_RNN.reshape((-1,))
# # vy = valid_y.reshape((-1,))

# mse_RNN = mean_squared_error(vy, py_RNN)
# print(mse_RNN)

# plt.plot(vy, color = 'red', label = 'Real Stock Price')  # 紅線表示真實股價
# plt.plot(py_RNN, color = 'blue', label = 'Predicted Stock Price')  # 藍線表示預測股價
# plt.title('Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('tock Price')
# plt.legend()
# plt.show()

"""BDLSTM"""

# p_model_BDLSTM = BDLSTM_model(train_x.shape)
# p_model_BDLSTM.load_weights("weights_best.hdf5")

p_model_BDLSTM = keras.models.load_model("weights_best.hdf5")

predict_BDLSTM = p_model_BDLSTM.predict(valid_x)
py_BDLSTM = predict_BDLSTM.reshape((-1,))

vy

py_BDLSTM

vy_origin = scaler_c.inverse_transform(valid_y)

py_BDLSTM_origin = scaler_c.inverse_transform(predict_BDLSTM)

# vy_origin = scaler_c.inverse_transform(vy)
# py_BDLSTM_origin = scaler_c.inverse_transform(py_BDLSTM)

mse_BDLSTM = mean_squared_error(vy, py_BDLSTM)
print(mse_BDLSTM)

mse_BDLSTM = mean_squared_error(vy_origin, py_BDLSTM_origin)
print(mse_BDLSTM)

plt.plot(vy_origin, color = 'red', label = 'Real Stock Price')  # 紅線表示真實股價
plt.plot(py_BDLSTM_origin, color = 'blue', label = 'Predicted Stock Price')  # 藍線表示預測股價
plt.title('Taiwan, MSE = ' + str(mse_BDLSTM))
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

np.save('BDLSTM', py_BDLSTM_origin)

np.save('real', vy_origin)