import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("survey_results_public.csv", header=0,
                 usecols=['YearsCode', 'Age', 'Age1stCode'])
df = df.dropna()
df.loc[df['YearsCode'] == 'Less than 1 year'] = 0
df.loc[df['YearsCode'] == 'More than 50 years'] = 51
df.loc[df['Age1stCode'] == 'Younger than 5 years'] = 0
df.loc[df['Age1stCode'] == 'Older than 85'] = 86
df['YearsCode'] = df['YearsCode'].astype("float64")
df['Age1stCode'] = df['Age1stCode'].astype('float64')

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.describe()

# Dzielenie danych na zbiór treningowy i testowy
X1_train, X1_test, Y_train, Y_test = train_test_split(df[['Age']], df[['YearsCode']],
                                                      test_size=0.33, random_state=42)
X1_X2_train, X1_X2_test, Y_train, Y_test = train_test_split(df[['Age', 'Age1stCode']], df[['YearsCode']],
                                                            test_size=0.33, random_state=42)

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
