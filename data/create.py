from preprocessor import *
import pandas as pd
df = preprocess('data.xls')
df.to_csv('data.csv')
