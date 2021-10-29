import pandas as pd
from data_loader import *
from splits import *
from operations import *

def preprocess(file): 
  '''
  loading the data for preprocessing
  '''

  data = data_load(file)  # loading the file in a pandas dataframe

  data = drop_col(data, ['P','ff10','VV',"W'W'",'Td','DD'])   # dropping irrelevant columns

  '''
  Preparing the data to group by Date, by splitting the Local Time Column into Date and Time Column
  '''

  col = ['DateTime', 'temp', 'pressure', 'humidity','wind speed', 'weather', 'cloud']   # renaming the columns for easy access
  data = rename(data,col)

  col = ['Date','Time']
  old = 'DateTime'
  new_dat = split(data,old,col)   # splitting the DateTime column

  old = ['DateTime']
  datas = insert(data,new_dat,old)    # inserting the new columns into the dataframe

  '''
  Now grouping the dataframe by Date and reducing it to one value per day
  '''
  '''
  Taking the mode value for all fields except temperature
  '''
  '''
  For temperature, we shall consider the maximum and minimum value per day and insert them as 2 separate columns
  '''

  df = mode(datas, 'Date')

  date = unique(datas, 'Date', 'Date')    # finding unique date value for insertion

  maxtemp = max(datas, 'Date', 'temp', 'maxtemp')   # max temp per day
  mintemp = min(datas, 'Date', 'temp', 'mintemp')   # min temp per day

  '''
  For wind speed, we shall consider the mea value during the day
  '''

  mean_wind = mean(datas, 'Date', 'wind speed', 'mean wind speed')    # mean wind speed during the day

  '''
  changing the index of the dataframe back to default value
  '''

  row = len(df.axes[0])
  df.index = range(row)

  '''
  inserting the relevant values in the new dataframe df
  '''

  df = insert(df,date)
  df = insert(df,maxtemp,None,2)
  df = insert(df,mintemp,None,3)
  df = insert(df,mean_wind,None,7)

  df = drop_col(df, ['Time', 'temp', 'wind speed'])  # dropping irrelevant columns

  return pd.DataFrame(df)