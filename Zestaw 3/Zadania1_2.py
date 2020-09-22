import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# input data
df_names = ['Occupancy', 'Date', 'Temperature', 'Humidity',
            'Light', 'CO2', 'HumidityRatio']
df = pd.read_csv('train.tsv', sep='\t', names=df_names)
df_column_names = ['Date', 'Temperature', 'Humidity', 'Light',
                   'CO2', 'HumidityRatio']
X_test = pd.read_csv('test.tsv', sep='\t', names=df_column_names, usecols=['Light'])
df_results = pd.read_csv('results.tsv', sep='\t', names=['y'])

df = df.dropna()
df.describe()

# logistic regression classifier on one independent variable
clf = LogisticRegression()
X_train = df[['Light']]
y_train = df.Occupancy
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
clf_accuracy = sum(y_train == y_train_pred) / len(df)
print("Training set accuracy for logisitic regression model "
      + "on Light variable:\n" + str(clf_accuracy))

df_results['y'] = df_results['y'].astype('category')
y_true = df_results['y']
y_test_pred = clf.predict(X_test)
clf_test_accuracy = accuracy_score(y_true, y_test_pred)
print('Accuracy on test dataset (full model): ' + str(clf_test_accuracy))

df = pd.DataFrame(y_test_pred)
df.to_csv('out.tsv', index=False, header=False)
