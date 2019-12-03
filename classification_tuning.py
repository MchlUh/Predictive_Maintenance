import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest

df_train = pd.read_csv('feature_engineered_train.csv').set_index('id')
df_test = pd.read_csv('feature_engineered_test.csv').set_index('id')

df_train['label'] = df_train.time_to_failure.apply(lambda x: 1 if x <= 50 else 0)
df_test['label'] = df_test.time_to_failure.apply(lambda x: 1 if x <= 50 else 0)
df_train, df_test = df_train.drop('time_to_failure', axis=1), df_test.drop('time_to_failure', axis=1)

X, y = df_train.drop('label', axis=1), df_train.label

kbest = SelectKBest(k=10)
kbest.fit(X, y)
selected_features = [X.columns[i] for i in range(X.shape[1]) if kbest.get_support()[i]==True]
X = X[selected_features]

log_reg = LogisticRegression()
log_reg.fit(X, y)

X_test, y_test = df_test.drop('label', axis=1), df_test.label
X_test = X_test[selected_features]
y_pred = log_reg.predict(X_test)

print(classification_report(y_test, y_pred))
