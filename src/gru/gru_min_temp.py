import pandas as pd
import numpy as np
import datetime

from sklearn import preprocessing

from keras.layers import Dense, Dropout, GRU

df = pd.read_csv('/data/data.csv')

le = preprocessing.LabelEncoder()
df['weather'] = le.fit_transform(df['weather'])
origin = datetime.datetime(2017,4,1)
for i in range(len(list(df['Date']))):
  df['Date'][i] = (datetime.datetime.strptime(df['Date'][i], '%d.%m.%Y') - origin).days

x2 = df[["Date", "pressure", "humidity", "mean wind speed", "weather"]].values
y2 = df["mintemp"].values

scaler = preprocessing.StandardScaler()
x2 = scaler.fit_transform(x2)
x2 = x2.reshape(-1, 1, 5)

regressor = Sequential()
regressor.add(GRU(6, input_shape=(1,5), return_sequences=True))
regressor.add(Dense(4, activation="relu"))
regressor.add(Dense(1, activation="linear"))
regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.fit(x2,y2,epochs=100,batch_size=10)
model.save('/model/gru_model_mintemp.h5')
