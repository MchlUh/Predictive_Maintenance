import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, RFECV
from yellowbrick.classifier import ClassificationReport
from yellowbrick.features import parallel_coordinates

df_train = pd.read_csv('feature_engineered_train.csv').set_index('id')
df_test = pd.read_csv('feature_engineered_test.csv').set_index('id')

df_train['label'] = df_train.time_to_failure.apply(lambda x: 1 if x <= 30 else 0)
df_test['label'] = df_test.time_to_failure.apply(lambda x: 1 if x <= 30 else 0)
df_train, df_test = df_train.drop('time_to_failure', axis=1), df_test.drop('time_to_failure', axis=1)


X, y = df_train.drop('label', axis=1), df_train.label
X_test, y_test = df_test.drop('label', axis=1), df_test.label


# Recursive feature elimination
# Very time consuming but picks the best features for the model
# Check sklearn RFECV docs
rfecv = RFECV(LogisticRegression(), scoring='f1')
rfecv.fit(X, y)
selected_features_rfe = [X.columns[i] for i in range(X.shape[1]) if rfecv.get_support()[i]==True]
print('selected RFE features:', selected_features_rfe)

# Select k most relevant features
# Play around with K to see which number of features is best
kbest = SelectKBest(k=10)
kbest.fit(X, y)
selected_features_kbest = [X.columns[i] for i in range(X.shape[1]) if kbest.get_support()[i]==True]
print('selected kbest features:', selected_features_kbest)


X = X[selected_features_rfe]
X_test = X_test[selected_features_rfe]


g = parallel_coordinates(X, y)
g = classification_report(LogisticRegression(), X, y)

model = LogisticRegressionCV()
visualizer = ClassificationReport(model)

visualizer.fit(X, y)
visualizer.score(X_test, y_test)
visualizer.show()

log_reg = LogisticRegressionCV()
log_reg.fit(X, y)
y_pred = log_reg.predict(X_test)

print(classification_report(y_test, y_pred))
