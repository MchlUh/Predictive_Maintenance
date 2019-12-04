import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, RFECV, RFE, SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import datetime
from sklearn.svm import SVR
from matplotlib import pyplot as plt

df_train = pd.read_csv('feature_engineered_train.csv').set_index('id')
df_test = pd.read_csv('feature_engineered_test.csv').set_index('id')

X, y = df_train.drop('time_to_failure', axis=1), df_train.time_to_failure


# Recursive feature elimination
# Very time consuming but picks the best features for the model
# Check sklearn RFECV docs
rfecv_lm = RFECV(LinearRegression(), scoring='neg_mean_absolute_error', cv=5)
rfecv_lm.fit(X, y)
selected_features_rfecv_lm = [X.columns[i] for i in range(X.shape[1]) if rfecv_lm.get_support()[i]]
print('selected RFECV features:', selected_features_rfecv_lm)

# --> Keeps 167 features out of 170.  Not helpfull.
# Also, do not account for interaction between features as uses linear regression.


# Select k most relevant features
# Play around with K to see which number of features is best
# Statistical Analysis of features.
kbest = SelectKBest(k=100)
kbest.fit(X, y)
selected_features_kbest = [X.columns[i] for i in range(X.shape[1]) if kbest.get_support()[i]==True]
print('selected kbest features:', selected_features_kbest)


# Much more advanced feature selection, using a Recursive feature elimination based on a RandomForest Regressor.
# Criterion mae is used.
tic = datetime.datetime.now()
rf = RandomForestRegressor(n_estimators=10, criterion='mae', max_depth=7, verbose=2, n_jobs=-1)
rf.fit(X, y)
toc = datetime.datetime.now()
print(toc-tic)
feat_imp = pd.DataFrame(X.columns, columns=['feat_name'])
feat_imp['imp'] = rf.feature_importances_
feat_imp.to_csv("rf_feat_importance.csv")
feat_imp = pd.read_csv("rf_feat_importance.csv")
feat_imp.sort_values(by='imp', ascending=False, inplace=True)

plt.plot(feat_imp.imp.values.cumsum())
plt.grid()
plt.show()

plt.plot(feat_imp.imp.values[5:].cumsum())
plt.grid()
plt.show()


# Train LinearRegression on RFE features
lin_reg = LinearRegression()
lin_reg.fit(X[selected_features_rfecv_lm], y)

X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
y_pred = lin_reg.predict(X_test[selected_features_rfecv_lm])
print('Linear Regression with RFE feature selection', mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred))

cv_scores_lin_reg = cross_val_score(lin_reg, X, y, scoring='mae')

# Train LinearRegression on RFE features
lin_reg = LinearRegression()
lin_reg.fit(X[selected_features_rfecv_lm], y)

X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
y_pred = lin_reg.predict(X_test[selected_features_rfecv_lm])
print('Linear Regression with RFE feature selection', mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred))

cv_scores_lin_reg = cross_val_score(lin_reg, X, y, scoring='mae')

# Train RandomForestRegressor on 50 most important features
n_est = 50
crit = 'mae'
max_depth = 7

rf = RandomForestRegressor(n_estimators=n_est, criterion=crit, max_depth=max_depth)
rf.fit(X[feat_imp.feat_name.values[:50]], y)

X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
y_pred = rf.predict(X_test[selected_features_rfe_rf_100])

print('Linear Regression with RFE feature selection', mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred))



# Train SVM on Best features for RandomForest
kernel = 'sigmoid'
gamma = 'scale'
C = 1
svr = SVR(kernel=kernel, gamma=gamma, C=C)
cv_scores_svr = cross_val_score(svr, X, y, scoring='mae')

X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
y_pred = lin_reg.predict(X_test[selected_features_kbest])

print('Linear Regression with kbest feature selection', mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred))


