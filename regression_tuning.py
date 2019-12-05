import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNetCV, ElasticNet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, RFECV, RFE, SelectFromModel
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import datetime
from sklearn.svm import SVR
from matplotlib import pyplot as plt
from plots import plot_error_repartition
import csv
import json

df_train = pd.read_csv('feature_engineered_train.csv').set_index('id')
df_test = pd.read_csv('feature_engineered_test.csv').set_index('id')

X, y = df_train.drop('time_to_failure', axis=1), df_train.time_to_failure


# Recursive feature elimination
# Very time consuming but picks the best features for the model
# Check sklearn RFECV docs
rfe = RFECV(LinearRegression(), scoring='neg_mean_absolute_error', cv=5)
rfe.fit(X, y)
selected_features_rfe = [X.columns[i] for i in range(X.shape[1]) if rfe.get_support()[i]]
print('selected RFECV features:', selected_features_rfe)

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
# tic = datetime.datetime.now()
# rf = RandomForestRegressor(n_estimators=10, criterion='mae', max_depth=7, verbose=2, n_jobs=-1)
# rf.fit(X, y)
# toc = datetime.datetime.now()
# print(toc-tic)
# feat_imp = pd.DataFrame(X.columns, columns=['feat_name'])
# feat_imp['imp'] = rf.feature_importances_
# feat_imp.to_csv("rf_feat_importance.csv")
feat_imp = pd.read_csv("rf_feat_importance.csv")
feat_imp.sort_values(by='imp', ascending=False, inplace=True)

plt.plot(feat_imp.imp.values.cumsum())
plt.grid()
plt.show()

plt.plot(feat_imp.imp.values[25:].cumsum())
plt.grid()
plt.show()

selected_features_rf_50 = feat_imp.feat_name.values[:50]
#
#############
#######################################################################################################
# DONE WITH FEATURE SELECTION --> GOING TO MODEL SELECTION ###################################
#######################################################################################################
############
#
# Train LinearRegression on RFE features
lin_reg = LinearRegression()
lin_reg.fit(X[selected_features_rfe], y)

X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
y_pred_lm = lin_reg.predict(X_test[selected_features_rfe])
print('Linear Regression with RFE feature selection',
      mean_absolute_error(y_test, y_pred_lm), mean_squared_error(y_test, y_pred_lm))
# Linear Regression with RFE feature selection 14.142734642572869 319.7344583322316
cv_scores_lin_reg = cross_val_score(lin_reg, X[selected_features_rfe], y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
cv_scores_lin_reg
# --> Overfitting

plot_error_repartition(y_test, y_pred_lm)




# Train ElasticNet regression:
# l1_ratio = 1
# normalize = True
# n_jobs = -1
# max_iter = 100000
# cv = 5
# alpha = [0.00031142146133688865]
# lm_elasticCV = ElasticNetCV(l1_ratio=l1_ratio,
#                           normalize=normalize,
#                           n_jobs=n_jobs,
#                           max_iter=max_iter,
#                           cv=cv,
#                           alphas=alpha,
#                           verbose=2)
# lm_elasticCV.fit(X[selected_features_rfe], y)
# lm_elasticCV.alpha_
# lm_elasticCV.alphas_
# lm_elasticCV.l1_ratio_

# lm_elasticCV_params = lm_elasticCV.get_params()

lm_elasticCV_params = {'alpha': 0.00031142146133688865,
 'fit_intercept': True,
 'l1_ratio': 1,
 'normalize': True,
 'max_iter': 20000}

lm_elastic = ElasticNet().set_params(**lm_elasticCV_params)
lm_elastic.fit(X[selected_features_rfe], y)

y_pred_lm_elastic = lm_elastic.predict(X_test[selected_features_rfe])
print('ElasticNet with RFE feature selection',
      mean_absolute_error(y_test, y_pred_lm_elastic), mean_squared_error(y_test, y_pred_lm_elastic))
# ElasticNet with RFE feature selection 16.789633429226136 441.0482719856923

plot_error_repartition(y_test, y_pred_lm_elastic)

# cv_scores_elasticNet = cross_val_score(lm_elastic, X[selected_features_rfe], y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
# cv_scores_elasticNet : array([-13.44161491, -12.67828467, -15.41808165, -15.1937048 , -15.53008139])
# --> No more Overfitting

# Tye to fit on log of time
lm_elastic.fit(X[selected_features_rfe], y.apply(lambda x: np.log(x+1)))

y_pred_lm_elastic = lm_elastic.predict(X_test[selected_features_rfe])
y_pred_lm_elastic = np.exp(y_pred_lm_elastic)-1
print('ElasticNet with RFE feature selection',
      mean_absolute_error(y_test, y_pred_lm_elastic), mean_squared_error(y_test, y_pred_lm_elastic))

#######################################
######    #######
#    #    #
# ####    ###
#   ##    #

# Train RandomForestRegressor on 50 most important features
rf_params = {'n_estimators': [10, 30, 100],
             'criterion': ['mae'],
             'max_depth': [5, 7],
             'n_jobs': [-1],
             'min_impurity_decrease': [0, 0.1]}


grid_search_rf = GridSearchCV(RandomForestRegressor(),
                              rf_params,
                              scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'],
                              refit='neg_mean_absolute_error',
                              cv=5,
                              verbose=2,
                              n_jobs=-1)

grid_search_rf.fit(X[feat_imp.feat_name.values[:50]], y)

grid_searchCV_results_rf = grid_search_rf.cv_results_

grid_searchCV_results_rf = pd.DataFrame(grid_searchCV_results_rf)
grid_searchCV_results_rf.to_csv("grid_searchCV_results_rf_pandas.csv")


w = csv.writer(open("grid_searchCV_results_rf.csv", "w"))
for key, val in grid_searchCV_results_rf.items():
    print(key)
    w.writerow([key, val])


rf = RandomForestRegressor(n_estimators=n_est, criterion=crit, max_depth=max_depth, n_jobs=-1)
rf.fit(X[feat_imp.feat_name.values[:50]], y)

X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
y_pred_rf = rf.predict(X_test[feat_imp.feat_name.values[:50]])

print('RandomForest Regressor with 50 Best features',
      mean_absolute_error(y_test, y_pred_rf), mean_squared_error(y_test, y_pred_rf))

plot_error_repartition(y_test, y_pred_rf)

# Train SVM on Best 50 features:

#####  #        #  ######
#       #      #   #     #
####     #    #    ######
   #      #  #     #    ##
####       ##      #    ##
svr_params = {'kernel': ['sigmoid', 'poly'], 'gamma': ['scale'], 'C': [0.5, 1, 1.5]}
grid_search_svr = GridSearchCV(SVR(),
                               svr_params,
                               scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'],
                               refit='neg_mean_absolute_error',
                               cv=5,
                               verbose=2,
                               n_jobs=-1)

grid_search_svr.fit(X[feat_imp.feat_name.values[:50]], y)
grid_searchCV_results_svr = grid_search_svr.cv_results_

w = csv.writer(open("grid_searchCV_results_svr.csv", "w"))
for key, val in grid_searchCV_results_svr.items():
    print(key)
    w.writerow([key, val])


svr = SVR(kernel=kernel, gamma=gamma, C=C)
cv_scores_svr = cross_val_score(svr, X, y, scoring='mae')

X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
y_pred_svr = lin_reg.predict(X_test[feat_imp.feat_name.values[:50]])

print('SVR with 50 best feature', mean_absolute_error(y_test, y_pred_svr), mean_squared_error(y_test, y_pred_svr))
plot_error_repartition(y_test, y_pred_rf)


##      ##    ######
  ##  ##     #
    ##       #  ####
  ##  ##     #    ##
##      ##    ####



def huber_loss_train(y_true, y_pred):
    d = y_true - y_pred
    h = 1  #h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


# define a loss that is MSE if time to failure is small (<50), and MAE otherwise.
def custom_loss_train(y_true, y_pred):
    if y_true < 50:
        grad, hess = huber_loss_train(y_true, y_pred)
    else:
        grad = 2*(y_true - y_pred)
        hess = 2
    return grad, hess



xg_params = {'max_depth': 3,
             'learning_rate': 0.1,
             'n_estimators': 100,
             'verbosity': 2,
             'objective': custom_loss_train,
             'booster': 'gbtree',
             'tree_method': 'auto',
             'n_jobs': -1,
             'gamma': 0,
             'min_child_weight': 1,
             'max_delta_step': 0,
             'subsample': 1,
             'colsample_bytree': 1,
             'colsample_bylevel': 1,
             'colsample_bynode': 1,
             'reg_alpha': 0,
             'reg_lambda': 1,
             'scale_pos_weight': 1,
             'base_score': 0.5,
             'random_state': 0,
             'missing': None,
             'num_parallel_tree': 1,
             'importance_type': 'gain'}


xgb_reg = xgb.XGBRegressor(**xg_params)

xgb_reg.fit(X[feat_imp.feat_name.values[:50]], y, verbose=2)

# xgb_reg.fit(X_train, y_train,
#         eval_set=[(X_train, y_train), (X_test, y_test)],
#         eval_metric='logloss',
#         verbose=True)

# evals_result = xgb_reg.evals_result()
xgb_reg.save_model('xgb1.model')

xgb.plot_importance(xgb_reg)

X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
y_pred_xgb = lin_reg.predict(X_test[feat_imp.feat_name.values[:50]])

print('SVR with 50 best feature', mean_absolute_error(y_test, y_pred_svr), mean_squared_error(y_test, y_pred_svr))
plot_error_repartition(y_test, y_pred_rf)