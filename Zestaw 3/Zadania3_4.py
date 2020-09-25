import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("survey_results_public.csv", header=0)
df = df.dropna()
df['under_30'] = np.where(df['Age'] <= 30, 1, 0)
df.loc[df['YearsCode'] == 'Less than 1 year'] = 0
df.loc[df['YearsCode'] == 'More than 50 years'] = 51
df['YearsCode'] = df['YearsCode'].astype("float64")

df.describe()
df.corr()

# logistic regression classifier on one independent variable
clf = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(df[['YearsCode']], df[['under_30']],
                                                    test_size=0.33, random_state=42)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_train = y_train['under_30'].to_numpy()
clf_accuracy = sum(y_train == y_train_pred) / len(df)
print("Training set accuracy for logisitic regression model "
      + "on Light variable:\n" + str(clf_accuracy))

conf_matrix = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = conf_matrix.ravel()
print(tn, fp, fn, tp)
sensitivity1 = conf_matrix[0, 0]/(conf_matrix[0, 0]+conf_matrix[0, 1])
print('Sensitivity : ', sensitivity1)
specificity1 = conf_matrix[1, 1]/(conf_matrix[1, 0]+conf_matrix[1, 1])
print('Specificity : ', specificity1)
print("Residual sum of squares: %.2f"
      % mean_squared_error(y_test, clf.predict(X_test)))
