import pandas as pd
import numpy as np
from sklearn import metrics
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pickle
import datetime
from sklearn.preprocessing import StandardScaler

html_temp = '''
    <div style = "background-color: rgba(25,25,112,0.0); padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Weather Predictor</h1></center>
    
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)

ann_model_maxtemp = tf.keras.models.load_model('model/ann_model_maxtemp.h5')
ann_model_mintemp = tf.keras.models.load_model('model/ann_model_mintemp.h5')

gru_model_maxtemp = tf.keras.models.load_model('model/gru_model_maxtemp.h5')
gru_model_mintemp = tf.keras.models.load_model('model/gru_model_mintemp.h5')

rf_model_maxtemp = pickle.load(open('model/rf_model_maxtemp.pkl','rb'))
rf_model_mintemp = pickle.load(open('model/rf_model_mintemp.pkl','rb'))

svr_model_maxtemp = pickle.load(open('model/svr_model_maxtemp.pkl','rb'))
svr_model_mintemp = pickle.load(open('model/svr_model_mintemp.pkl','rb'))

model = st.selectbox("Please Select your model from the following options",("Please Select","Artificial Neural Networks","Support Vector Machine","Random Forest Regressor","Gated Recurrent Units"))

data = []
data.append(st.sidebar.date_input('Enter Date'))
data.append(st.sidebar.slider('Pressure',700,800,760))
data.append(st.sidebar.number_input('Humidity'))
data.append(st.sidebar.number_input('Wind Speed'))
data.append(st.sidebar.selectbox("Weather Condition",("Please Select","Fog","Haze","Light Rain","Mist","Rain","Smoke")))

#st.success(data[0])
origin = datetime.datetime(2017,4,1)
data[0] = (datetime.datetime.strptime(str(data[0]), '%Y-%m-%d') - origin).days

if data[4] == 'Fog':
    data[4] = 0
elif data[4] == 'Haze':
    data[4] = 1
elif data[4] == 'Light Rain':
    data[4] = 2
elif data[4] == 'Mist':
    data[4] = 3
elif data[4] == 'Rain':
    data[4] = 4
elif data[4] == 'Smoke':
    data[4] = 5

scaler = StandardScaler()
data = scaler.fit_transform([data])

#st.write(data.shape)

if st.button('Predict'):

    if model == "Artificial Neural Networks":
        max = ann_model_maxtemp.predict([data])
        min = ann_model_mintemp.predict([data])
        st.info('Maximum Temperature: {} \n Minimum Temperature: {}'.format(max[0][0],min[0][0]))
    
    if model == "Support Vector Machine":
        data = data.tolist()
        st.write(data)
        max = svr_model_maxtemp.predict(data)
        min = svr_model_mintemp.predict(data)
        st.info('Maximum Temperature: {}\nMinimum Temperature: {}'.format(max[0],min[0]))
    
    if model == "Random Forest Regressor":
        max = rf_model_maxtemp.predict(pd.DataFrame(columns=["Date", "pressure", "humidity", "mean wind speed", "weather"],data=np.array(data).reshape(1,5)))
        min = rf_model_mintemp.predict(pd.DataFrame(columns=["Date", "pressure", "humidity", "mean wind speed", "weather"],data=np.array(data).reshape(1,5)))
        st.info('Maximum Temperature: {}\nMinimum Temperature: {}'.format(max[0],min[0]))

    if model == "Gated Recurrent Units":
        max = gru_model_maxtemp.predict([data])
        min = gru_model_mintemp.predict([data])
        st.info('Maximum Temperature: {}\nMinimum Temperature: {}'.format(max[0][0],min[0][0]))
