import pandas as pd
from statistics import stdev
import numpy as np

df = pd.read_csv("survey_results_public.csv", header=0,
                 usecols=['WorkWeekHrs', 'ConvertedComp', 'CompTotal', 'CodeRevHrs',
                          'YearsCode', 'Age', 'Age1stCode', 'Gender', 'Hobbyist'])
df = df.dropna()
df.loc[df['YearsCode'] == 'Less than 1 year'] = 0
df.loc[df['YearsCode'] == 'More than 50 years'] = 51
df.loc[df['Age1stCode'] == 'Younger than 5 years'] = 0
df.loc[df['Age1stCode'] == 'Older than 85'] = 86
df['YearsCode'] = df['YearsCode'].astype("float64")
df['Age1stCode'] = df['Age1stCode'].astype('float64')
print(df.corr(method='pearson'))

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df.corr(method='pearson'))
