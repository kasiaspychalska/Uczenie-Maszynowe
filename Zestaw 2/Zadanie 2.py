import pandas as pd

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

housing_map = {'Yes': 1, 'No': 0}
df['Hobbyist'] = df['Hobbyist'].map(housing_map)
df['Gender'] = df['Gender'].astype('str')
df = df[df['Gender'].isin(['Woman', 'Man'])]
df = pd.get_dummies(df, columns=['Gender'])
