import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

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

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.describe()

# Dzielenie danych na zbiór treningowy i testowy
X1_train = df[['Age']].head(len(df.index)-round(len(df.index)*0.1))
X1_test = df[['Age']].tail(round(len(df.index)*0.1))
Y_train = df['YearsCode'].head(len(df.index)-round(len(df.index)*0.1))
Y_test = df['YearsCode'].tail(round(len(df.index)*0.1))
X1_X2_train = df[['Age', 'Age1stCode']].head(len(df.index)-round(len(df.index)*0.1))
X1_X2_test = df[['Age', 'Age1stCode']].tail(round(len(df.index)*0.1))
X3_train = df[['Age', 'Age1stCode', 'Gender_Man', 'Gender_Woman', 'Hobbyist']].head(len(df.index)-round(len(df.index)*0.1))
X3_test = df[['Age', 'Age1stCode', 'Gender_Man', 'Gender_Woman', 'Hobbyist']].tail(round(len(df.index)*0.1))

regr1 = linear_model.LinearRegression()
regr1.fit(X1_train, Y_train)

print('Coefficients: \n', regr1.coef_)
print("Residual sum of squares: %.2f"
      % mean_squared_error(Y_test, regr1.predict(X1_test)))

regr2 = linear_model.LinearRegression()
regr2.fit(X1_X2_train, Y_train)

print('Coefficients: \n', regr2.coef_)
print("Residual sum of squares: %.2f"
      % mean_squared_error(Y_test, regr2.predict(X1_X2_test)))

regr3 = linear_model.LinearRegression()
regr3.fit(X3_train, Y_train)

print('Coefficients: \n', regr3.coef_)
print("Residual sum of squares: %.2f"
      % mean_squared_error(Y_test, regr3.predict(X3_test)))
