import pandas as pd 
import os

dir = 'data/air_quality/air_quality/'
files = os.listdir(dir)

df = pd.read_csv(dir + str(files[0]))

print(df.shape)
print(df.isna().sum())
print(df.dtypes)
print(df.loc[:, df.dtypes!=object].shape)

print(df.std())
