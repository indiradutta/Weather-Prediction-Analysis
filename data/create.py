from preprocessor import *
import pandas as pd
df = preprocess('data.xlsx')
df.to_csv('data.csv')
