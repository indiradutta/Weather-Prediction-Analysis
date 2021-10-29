import pandas as pd
import numpy as np
import datetime

from keras.models import Sequential
from keras.layers import Dense

from sklearn import preprocessing

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

model = Sequential()
model.add(Dense(8, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x2,y2, epochs=100, batch_size=10)
model.save('/cmodel/ann_model_mintemp.h5')
