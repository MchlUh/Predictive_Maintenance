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
import warnings
from plots import plot_classification_report

warnings.simplefilter(action='ignore', category=FutureWarning)

df_train = pd.read_csv('feature_engineered_train.csv').set_index('id')
df_test = pd.read_csv('feature_engineered_test.csv').set_index('id')

time_to_failure_test = df_test.time_to_failure

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


# Feature selection with recursive feature elimination. Uncomment to perform again
# rfecv = RFECV(LogisticRegression(), scoring='f1')
# rfecv.fit(X, y)
# selected_features_rfe_log_reg = [X.columns[i] for i in range(X.shape[1]) if rfecv.get_support()[i]==True]
# print('selected RFE features for logistic regression:', selected_features_rfe_log_reg)


# Last result of Recursive feature elimination
selected_features_rfe_log_reg = ['sensor4', 'sensor8', 'sensor9', 'sensor11', 'sensor17', 'sensor17_rolling_mean_20',
                                 'sensor20_rolling_mean_20', 'sensor4_rolling_max_20', 'sensor8_rolling_max_20',
                                 'sensor11_rolling_max_20', 'sensor17_rolling_max_20', 'sensor12_rolling_min_20',
                                 'sensor17_rolling_energy_20', 'sensor20_rolling_energy_20',
                                 'sensor4_rolling_sum_of_changes_20', 'sensor7_rolling_sum_of_changes_20',
                                 'sensor11_rolling_sum_of_changes_20', 'sensor12_rolling_sum_of_changes_20',
                                 'sensor14_rolling_sum_of_changes_20', 'sensor20_rolling_sum_of_changes_20',
                                 'sensor21_rolling_sum_of_changes_20', 'sensor2_rolling_variance_20',
                                 'sensor14_rolling_variance_20', 'sensor2_time_reversal_asymmetry_window_20_lag_5',
                                 'sensor4_time_reversal_asymmetry_window_20_lag_5',
                                 'sensor7_time_reversal_asymmetry_window_20_lag_5',
                                 'sensor12_time_reversal_asymmetry_window_20_lag_5',
                                 'sensor20_time_reversal_asymmetry_window_20_lag_5',
                                 'sensor21_time_reversal_asymmetry_window_20_lag_5',
                                 'sensor8_log', 'sensor9_log', 'sensor12_log']


log_reg = LogisticRegressionCV(class_weight={0: 1, 1: 2})
log_reg.fit(X[selected_features_rfe_log_reg], y)
y_pred = log_reg.predict(X_test[selected_features_rfe_log_reg])
print('Logistic regression performance on RFE features: \n', classification_report(y_test, y_pred))
plot_classification_report(y_test, y_pred, time_to_failure_test, name='Logistic Regression')


# KNN
knn_pipe = Pipeline(steps=[('lda', LinearDiscriminantAnalysis()),
                           ('knn', KNeighborsClassifier())])
grid_params = {
    'knn__n_neighbors': range(5, 20)
}
knn_grid = RandomizedSearchCV(knn_pipe, grid_params, n_iter=10)
knn_grid.fit(X[selected_features_rfe_log_reg], y)
print('KNN with KBest best params : ', knn_grid.best_params_)
y_pred = knn_grid.predict(X_test[selected_features_rfe_log_reg])
print('KNN performance on Kbest features: \n', classification_report(y_test, y_pred))
plot_classification_report(y_test, y_pred, time_to_failure_test, name='KNN')

# XGBOOST
xgb_class = xgb.XGBClassifier(max_depth=3, subsample=0.8, colsample_bytree=0.8, n_estimators=300)
xgb_class.fit(X[selected_features_rfe_log_reg], y)
y_pred = xgb_class.predict(X_test[selected_features_rfe_log_reg])
print('XGBoost performance on RFE features: \n', classification_report(y_test, y_pred))
plot_classification_report(y_test, y_pred, time_to_failure_test, name='XGBClassifier')


