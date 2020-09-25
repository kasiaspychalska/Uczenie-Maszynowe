import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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
X_train = df[['YearsCode']]
y_train = df[['under_30']]
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
