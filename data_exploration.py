import pandas as pd 
import os

dir = 'data/breast_cancer/'
files = os.listdir(dir)

df = pd.read_csv(dir + str(files[0]), names=list(range(32)))
# df = df.select_dtypes(include=['float64', 'int64'])# df.loc[:, df.dtypes != object]

print(df.shape)
print(df.isna().sum())
print(df.dtypes)
print(df.loc[:, df.dtypes!=object].shape)

print(df.std())
