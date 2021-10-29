import pandas as pd
import data_loader

def rename(dataframe, new_columns):
  '''
  new_columns should be passed as a list of column names to be renamed as well as original column names that are not meant to be renamed
  '''

  dataframe.columns = new_columns
  return dataframe

def split(dataframe, old_column, new_columns):
  '''
  old_column is the column that should be splitted
  '''
  '''
  new_columns should be passed as a list of column names to be renamed
  '''
  new_dataframe = dataframe[old_column].str.split(" ", n = 1, expand = True)
  new_dataframe = rename(new_dataframe, new_columns)
  return new_dataframe

def insert(dataframe, new_dataframe, old_column = None, location = 0):
  '''
  dataframe is the original dataframe where insertion should be performed
  '''
  '''
  new_dataframe has the fields that should be inserted in dataframe
  '''
  '''
  old column should be entered as a list of columns to be replaced
  '''
  '''
  location mentions where the new column must be inserted
  '''
  
  columns = new_dataframe.columns.tolist()
  for i in columns:
    values = new_dataframe[i].tolist()
    dataframe.insert(loc = location,
                                column = i,
                                value = values)
    location+=1
  if old_column is not None:
    dataframe = data_loader.drop_col(dataframe, old_column)
  return dataframe