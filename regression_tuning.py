import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, RFECV

df_train = pd.read_csv('feature_engineered_train.csv').set_index('id')
df_test = pd.read_csv('feature_engineered_test.csv').set_index('id')

X, y = df_train.drop('time_to_failure', axis=1), df_train.time_to_failure


# Recursive feature elimination
# Very time consuming but picks the best features for the model
# Check sklearn RFECV docs
rfecv = RFECV(LinearRegression(), scoring='neg_mean_absolute_error')
rfecv.fit(X, y)
selected_features_rfe = [X.columns[i] for i in range(X.shape[1]) if rfecv.get_support()[i]==True]
print('selected RFE features:', selected_features_rfe)


# Select k most relevant features
# Play around with K to see which number of features is best
kbest = SelectKBest(k=10)
kbest.fit(X, y)
selected_features_kbest = [X.columns[i] for i in range(X.shape[1]) if kbest.get_support()[i]==True]
print('selected kbest features:', selected_features_kbest)


# Train model on RFE features
lin_reg = LinearRegression()
lin_reg.fit(X[selected_features_rfe], y)

X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
y_pred = lin_reg.predict(X_test[selected_features_rfe])

print('Linear Regression with RFE feature selection', mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred))


# Train model on kbest features
lin_reg = LinearRegression()
lin_reg.fit(X[selected_features_kbest], y)

X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
y_pred = lin_reg.predict(X_test[selected_features_kbest])

print('Linear Regression with kbest feature selection', mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred))
