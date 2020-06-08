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

df = df.loc[(df['Age1stCode'] >= np.quantile(df.Age1stCode, 0.25) - stdev(df.Age1stCode)) & (df['Age1stCode'] <= np.quantile(df.Age1stCode, 0.75) + stdev(df.Age1stCode))]
print(df.corr(method='pearson'))
df = df.loc[(df['Age'] >= np.quantile(df.Age, 0.25) - stdev(df.Age)) & (df['Age'] <= np.quantile(df.Age, 0.75) + stdev(df.Age))]
print(df.corr(method='pearson'))

# Usuwając wartości poniżej 1 kwartyla i powyżej 3 (uwzględniając odchylenie standardowe)
# wzmacniamy korelację między kolumnami YearsCode i Age1stCode kosztem małego osłabienia korelacji między kolumnami YearsCode i Age.
# W przypadku usunięcia odchyleń tylko dla kolumny Age1stCode wzmocnilibyśmy korelację między kolumnami YearsCode i Age
# ale tylko nieznacznie zostałaby wzmocniona korelacja między drugimi zmiennymi.
