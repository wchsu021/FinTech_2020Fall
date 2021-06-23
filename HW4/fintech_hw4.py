import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from mpl_finance import candlestick_ohlc
from mpl_finance import volume_overlay

from talib import abstract

from matplotlib import dates as mpdates
from matplotlib import ticker as mticker
from matplotlib.dates import DateFormatter
import datetime as dt

import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Activation, Dense, Dropout, Input
from keras.optimizers import Adam


def plot_stock_price(data):
	
	data = data.rename(columns = {'Open': 'open','High': 'high', 'Low': 'low', 'Close': 'close'}, inplace = False)
	
	Ma_10 = abstract.MA(data, timeperiod=10, matype=0)
	Ma_30 = abstract.MA(data, timeperiod=30, matype=0)

	fig = plt.figure(facecolor='white',figsize=(15,10))
	ax1 = plt.subplot2grid((6,4), (0,0),rowspan=4, colspan=4, facecolor='w')
	data['Date'] = pd.to_datetime(data['Date']) 
	data['Date'] = data['Date'].map(mpdates.date2num) 
	
	# fig, ax1 = plt.subplots() 
	candlestick_ohlc(ax1, data.values,width=0.6,colorup='red',colordown='green')
	Label1 = 'MA_10'
	Label2 = 'MA_30'
	ax1.plot(data.Date.values,Ma_10,label=Label1, linewidth=1.5)
	ax1.plot(data.Date.values,Ma_30,label=Label2, linewidth=1.5)
	ax1.legend(loc='upper right', ncol=2)
	# ax1.grid(True, color='black')
	ax1.xaxis.set_major_formatter(mpdates.DateFormatter('%Y-%m-%d'))
	ax1.yaxis.label.set_color("black")
	ax1.tick_params(axis='y', colors='black')
	ax1.tick_params(axis='x', colors='black')
	plt.ylabel('Stock price and Volume')
	# plt.suptitle('Stock Code:0000',color='black',fontsize=16)
		
	ax2 = ax1.twinx()
	ax2 = plt.subplot2grid((6,4), (4,0), sharex=ax1, rowspan=1, colspan=4, facecolor='white')
	# df = KD(data)
	# df = abstract.STOCH(data)
	kd = abstract.STOCH(data)
	# abstract.STOCH(data).plot(figsize=(16,8))
	# print(abstract.STOCH(data).tail(10))
	ax2.plot(data.Date.values, kd.slowk, label = 'K')
	ax2.plot(data.Date.values, kd.slowd, label = 'D')
	ax2.legend(loc='upper right', ncol=2)

	plt.ylabel('KD Value', color='black')

	ax3 = ax1.twinx()
	ax3 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4, facecolor='white')
	plt.ylabel('Volume', color='black')
	# print(data['Volume'])
	# ax3.bar(data.Date.values, data['Volume'])
	volume_overlay(ax3, data['open'], data['close'], data['Volume'], colorup='r', colordown='g')

	

	plt.tight_layout()
	# plt.savefig("stock_2019.jpg")
	plt.show()

def data_preprocess(data):

	data = data.rename(columns = {'Open': 'open','High': 'high', 'Low': 'low', 'Close': 'close'}, inplace = False)
	
	MA_10 = abstract.MA(data, timeperiod=10, matype=0)
	MA_30 = abstract.MA(data, timeperiod=30, matype=0)
	KD = abstract.STOCH(data)

	# print(data.head(40))
	# print(MA_10.head(40))	#NaN:0-8
	# print(MA_30.head(40))	#NaN:0-28
	# print(KD.head(40))	#NaN:0-7

	data['MA_10'] = MA_10
	data['MA_30'] = MA_30
	data['K'] = KD['slowk']
	data['D'] = KD['slowd']

	data = data.rename(columns = {'open': 'Open','high': 'High', 'low': 'Low', 'close': 'Close'}, inplace = False)
	# print(data)
	data = data.dropna()

	output ='feature_2020.csv'
	data.to_csv(output, index=False )
	# print(data)
	# data["concavity_mean"]=((data["concavity_mean"]-data["concavity_mean"].min())/(data["concavity_mean"].max()-data["concavity_mean"].min()))*20
	cols = list(data)
	# print(cols)
	cols.pop(0)
	# print(cols)
	for col in cols:
		data[col]=(data[col]-data[col].min())/(data[col].max()-data[col].min())
	# normalized_data=(data-data.min())/(data.max()-data.min())
	# print(data)
	return data
	pass

def gen_input_data(data, time_step = 30, future_day = 1):

	data = data.drop(["Date"], axis = 1)
	train_x, train_y = [], []
	for i in range(data.shape[0]-time_step-future_day):
		train_x.append(np.array(data.iloc[i:i+time_step]))
		train_y.append(np.array(data.iloc[i+time_step:i+time_step+future_day]["Close"]))
	return np.array(train_x), np.array(train_y)

	pass

def SimpleRNN_model(shape):

	batch_input_shape = (None, shape[1], shape[2])
	print(batch_input_shape)
	model = Sequential()
	model.add(SimpleRNN(32, batch_input_shape = batch_input_shape, unroll=True))
	model.add(Dropout(0.2))
	model.add(Dense(1))
	model.add(Activation('softmax'))
	model.compile(loss = "mse", optimizer = "adam")
	# model.summary()

	return model
	pass

def LSTM_model(shape):

	model = Sequential()
	model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
	# output shape: (1, 1)
	model.add(Dense(1))
	model.compile(loss="mse", optimizer="adam")
	model.summary()
	return model
	'''
	model = Sequential()
	model.add(LSTM())
	model.add(Dense(1))
	model.compile(loss = "mse", optimizer = "adam")
	model.summary()

	return model
	'''
	pass

def rnn_model_test():

	batchs_size = 32
	inputs = Input(batch_shape=(batchs_size, 30, 10))
	RNN1 = SimpleRNN(units=128, activation='tanh', return_sequences=False, return_state=False)
	RNN1_output = RNN1(inputs)
	Dense1 = Dense(units=128, activation='relu')
	Dense2 = Dense(units=64, activation='relu')
	output = Dense(units=10, activation='softmax')
	rnn = Model(inputs=inputs, outputs=output)
	opt = Adam(lr=0.001, decay=1e-6)
	rnn.compile(optimizer=opt,
	              loss='mse',
	              metrics=['accuracy'])
	rnn.summary()
	return rnn

if __name__ == '__main__':
	
	file = 'S_P.csv'
	chart = pd.read_csv(file)
	chart_2019 = chart[(chart['Date'] >= "2019-01-01") & (chart['Date'] <= "2019-12-31")]

	chart_2020 = pd.read_csv('S_P_2020.csv')

	# print(chart_2019.head())
	# print(chart_2019.tail())

	# plot_stock_price(chart_2019)

	# print(chart)

	# input_data = data_preprocess(chart)
	input_data = data_preprocess(chart_2020)
	print(input_data)
	print("Finish Preprocess")

	# print("-"*40)
	# train_x, train_y = gen_input_data(input_data)
	# print(train_x[0])
	# print(len(train_x))
	# print("-*"*20)
	# print(train_y[0])
	# print(len(train_y))

	# print(train_x.shape)
	# print(train_y.shape)

	# print("Start train")

	# # model = SimpleRNN_model(train_x.shape)
	# # model = LSTM_model(train_x.shape)
	# # model.fit(train_x, train_y, epochs = 100, batch_size = 32)

	# model = rnn_model_test()
	# model.fit(train_x, train_y, epochs=1, batch_size = 32)
	# model.summary()


	

