# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 22:50:45 2018

@author: Niraj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds_train = pd.read_csv('Stock_Price_Train.csv')
ts_set = ds_train.iloc[:,1:2].values

# Feature Scaling Normalization

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
ts_scaled = sc.fit_transform(ts_set)

# ---RNN---

X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(ts_scaled[i-60:i,0])
    y_train.append(ts_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)
    
# Reshaping of data

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

rg = Sequential()

## Building RNN with LSTM

rg.add(LSTM(units=50, return_sequences=True, input_shape =(X_train.shape[1], 1)))
rg.add(Dropout(0.2))

rg.add(LSTM(units=50, return_sequences=True))
rg.add(Dropout(0.2))

rg.add(LSTM(units=50, return_sequences=True))
rg.add(Dropout(0.2))

rg.add(LSTM(units=50))
rg.add(Dropout(0.2))

rg.add(Dense(units=1))

rg.compile(optimizer='adam', loss='mean_squared_error')

rg.fit(X_train, y_train, epochs=10, batch_size=32)

# Checking on Test_set

ds_test = pd.read_csv('Stock_Price_Test.csv')
actual_price = ds_test.iloc[:,1:2].values

dataset = pd.concat((ds_train['Open'], ds_test['Open']),axis=0)
inputs =  dataset[len(dataset)-len(ds_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted = rg.predict(X_test)

predicted = sc.inverse_transform(predicted)

# Visualization

plt.plot(actual_price, color='red', label='Actual Stock Price')
plt.plot(predicted, color='blue', label='Predicted Stock Price')

plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()











