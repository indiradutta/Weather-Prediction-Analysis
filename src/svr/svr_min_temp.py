import pandas as pd
import numpy as np
import datetime

from sklearn import preprocessing
from sklearn.svm import SVR

df = pd.read_csv('/data/data.csv')

le = preprocessing.LabelEncoder()
df['weather'] = le.fit_transform(df['weather'])
origin = datetime.datetime(2017,4,1)
for i in range(len(list(df['Date']))):
  df['Date'][i] = (datetime.datetime.strptime(df['Date'][i], '%d.%m.%Y') - origin).days

x2 = df[["Date", "maxtemp", "pressure", "humidity", "mean wind speed", "weather"]].values
y2 = df["mintemp"].values

scaler = preprocessing.StandardScaler()
x2 = scaler.fit_transform(x2)

regressor = SVR(kernel='rbf')
regressor.fit(x2,y2)
filename = 'svr_model_mintemp.pkl'
pickle.dump(regressor, open(filename, 'wb'))