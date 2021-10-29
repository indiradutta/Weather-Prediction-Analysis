import pandas as pd

def unique(dataframe, column, new_column):
  '''
  column should be entered as a string mentioning from which column unique values should be extracted
  '''
  '''
  new_column should be entered as a string mentioning the name of the column contaning the unique values
  '''

  new_dataframe = pd.DataFrame(dataframe[column].unique().tolist(),columns=[new_column])
  return new_dataframe

def mode(dataframe, column):
  '''
  column should be entered as a string mentioning which column of the dataframe is to be used to group by
  '''

  new_dataframe = dataframe.groupby([column]).agg(lambda x:x.value_counts().index[0])
  return new_dataframe

def max(dataframe, column, req_column, new_column):
  '''
  column should be entered as a string mentioning which column of the dataframe is to be used to group by
  '''
  '''
  req_column should be entered as a string mentioning the name of the column whose max values are required
  '''
  '''
  new_column should be entered as a string mentioning the name of the column contaning the max values
  '''

  new_dataframe = pd.DataFrame(dataframe.groupby([column])[req_column].max().tolist(), columns = [new_column])
  return new_dataframe

def min(dataframe, column, req_column, new_column):
  '''
  column should be entered as a string mentioning which column of the dataframe is to be used to group by
  '''
  '''
  req_column should be entered as a string mentioning the name of the column whose min values are required
  '''
  '''
  new_column should be entered as a string mentioning the name of the column contaning the min values
  '''

  new_dataframe = pd.DataFrame(dataframe.groupby([column])[req_column].min().tolist(), columns = [new_column])
  return new_dataframe

def mean(dataframe, column, req_column, new_column):
  '''
  column should be entered as a string mentioning which column of the dataframe is to be used to group by
  '''
  '''
  req_column should be entered as a string mentioning the name of the column whose mean values are required
  '''
  '''
  new_column should be entered as a string mentioning the name of the column contaning the mean values
  '''

  new_dataframe = pd.DataFrame(dataframe.groupby([column])[req_column].mean().tolist(), columns = [new_column])
  return new_dataframe