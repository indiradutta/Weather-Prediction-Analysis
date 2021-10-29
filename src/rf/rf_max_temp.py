import pandas as pd
import numpy as np
import datetime

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

le = preprocessing.LabelEncoder()
df['weather'] = le.fit_transform(df['weather'])
origin = datetime.datetime(2017,4,1)
for i in range(len(list(df['Date']))):
  df['Date'][i] = (datetime.datetime.strptime(df['Date'][i], '%d.%m.%Y') - origin).days

x1 = df[["Date", "mintemp", "pressure", "humidity", "mean wind speed", "weather"]].values
y1 = df["maxtemp"].values

scaler = preprocessing.StandardScaler()
x1 = scaler.fit_transform(x1)

rf = RandomForestClassifier(n_estimators=10, criterion='gini')
rf.fit(x1,y1)
y1_pred = rf.predict(x1)
mean_squared_error(y1, y1_pred)