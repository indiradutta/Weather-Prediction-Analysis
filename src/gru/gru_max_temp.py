import pandas as pd
import numpy as np
import datetime

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from keras.layers import Dense, Dropout, GRU

le = preprocessing.LabelEncoder()
df['weather'] = le.fit_transform(df['weather'])
origin = datetime.datetime(2017,4,1)
for i in range(len(list(df['Date']))):
  df['Date'][i] = (datetime.datetime.strptime(df['Date'][i], '%d.%m.%Y') - origin).days

x1 = df[["Date", "mintemp", "pressure", "humidity", "mean wind speed", "weather"]].values
y1 = df["maxtemp"].values

scaler = preprocessing.StandardScaler()
x1 = scaler.fit_transform(x1)
x1 = x1.reshape(-1, 1, 6)

regressor = Sequential()
regressor.add(GRU(6, input_shape=(1,6), return_sequences=True))
regressor.add(Dense(4, activation="relu"))
regressor.add(Dense(1, activation="linear"))
regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.fit(x1,y1,epochs=100,batch_size=10)
model.save('/models/gru_model_maxtemp.h5')
