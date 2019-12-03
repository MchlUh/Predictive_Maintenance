import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df_train = pd.read_csv('feature_engineered_train.csv').set_index('id')
df_test = pd.read_csv('feature_engineered_test.csv').set_index('id')

X, y = df_train.drop('time_to_failure', axis=1), df_train.time_to_failure

log_reg = LinearRegression()
log_reg.fit(X, y)

X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
y_pred = log_reg.predict(X_test)

print(mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred))
