import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb

df_train = pd.read_csv('feature_engineered_train.csv').set_index('id')
df_test = pd.read_csv('feature_engineered_test.csv').set_index('id')

df_train['label'] = df_train.time_to_failure.apply(lambda x: 1 if x <= 30 else 0)
df_test['label'] = df_test.time_to_failure.apply(lambda x: 1 if x <= 30 else 0)
df_train, df_test = df_train.drop('time_to_failure', axis=1), df_test.drop('time_to_failure', axis=1)

X, y = df_train.drop('label', axis=1), df_train.label
X_test, y_test = df_test.drop('label', axis=1), df_test.label

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)


# Select k most relevant features
kbest = SelectKBest(k=10)
kbest.fit(X, y)
selected_features_kbest = [X.columns[i] for i in range(X.shape[1]) if kbest.get_support()[i]==True]
print('selected kbest features:', selected_features_kbest)


# LOGISTIC REGRESSION
log_reg = LogisticRegressionCV()
log_reg.fit(X[selected_features_kbest], y)
y_pred = log_reg.predict(X_test[selected_features_kbest])
print('Logistic regression performance on Kbest features: \n', classification_report(y_test, y_pred))


# Feature selection with recursive feature elimination
rfecv = RFECV(LogisticRegression(), scoring='f1')
rfecv.fit(X, y)
selected_features_rfe_log_reg = [X.columns[i] for i in range(X.shape[1]) if rfecv.get_support()[i]==True]
print('selected RFE features for logistic regression:', selected_features_rfe_log_reg)

log_reg = LogisticRegressionCV()
log_reg.fit(X[selected_features_rfe_log_reg], y)
y_pred = log_reg.predict(X_test[selected_features_rfe_log_reg])
print('Logistic regression performance on RFE features: \n', classification_report(y_test, y_pred))


# KNN
knn_pipe = Pipeline(steps=[('kbest', SelectKBest()),
                           ('lda', LinearDiscriminantAnalysis()),
                           ('knn', KNeighborsClassifier())])
grid_params = {
    'kbest__k': range(10, 50),
    'knn__n_neighbors': range(5, 20)
}
knn_grid = RandomizedSearchCV(knn_pipe, grid_params, n_iter=20)
knn_grid.fit(X, y)
print('KNN with KBest best params : ', knn_grid.best_params_)
y_pred = knn_grid.predict(X_test)
print('KNN performance on Kbest features: \n', classification_report(y_test, y_pred))


# XGBOOST
xgb_class = xgb.XGBRFClassifier(max_depth=3, subsample=0.8, colsample_bytree=0.8, n_estimators=300)
xgb_class.fit(X[selected_features_kbest], y)
y_pred = xgb_class.predict(X_test[selected_features_kbest])
print('XGBoost performance on Kbest features: \n', classification_report(y_test, y_pred))
