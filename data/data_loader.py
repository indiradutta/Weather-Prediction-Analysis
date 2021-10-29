import pandas as pd

def data_load(path):
  '''
  file should be of .xls format only
  '''

  df = pd.read_excel(path)
  return df

def drop_col(dataframe, fields):
  '''
  fields should be passed as a list of column names to be dropped from the passed dataframe
  '''

  df = dataframe.drop(fields, axis=1)
  return df