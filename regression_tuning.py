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
from plots import *
import csv
import json

df_train = pd.read_csv('feature_engineered_train.csv').set_index('id')
df_test = pd.read_csv('feature_engineered_test.csv').set_index('id')

X, y = df_train.drop('time_to_failure', axis=1), df_train.time_to_failure
X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure


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

y_pred_lm = lin_reg.predict(X_test[selected_features_rfe])
print('Linear Regression with RFE feature selection',
      mean_absolute_error(y_test, y_pred_lm), mean_squared_error(y_test, y_pred_lm))
# Linear Regression with RFE feature selection 14.142734642572869 319.7344583322316
cv_scores_lin_reg = cross_val_score(lin_reg, X[selected_features_rfe], y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
cv_scores_lin_reg
# --> Overfitting

plot_error_repartition(y_test, y_pred_lm, model_name='Linear Regression', save=True)



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
 'max_iter': 50000}

lm_elastic = ElasticNet().set_params(**lm_elasticCV_params)
lm_elastic.fit(X[selected_features_rfe], y)

y_pred_lm_elastic = lm_elastic.predict(X_test[selected_features_rfe])
print('ElasticNet with selected_features_rfe',
      mean_absolute_error(y_test, y_pred_lm_elastic), mean_squared_error(y_test, y_pred_lm_elastic))
# ElasticNet with RFE feature selection 16.789633429226136 441.0482719856923

plot_error_repartition(y_test, y_pred_lm_elastic, model_name='ElasticNet', save=True)

# cv_scores_elasticNet = cross_val_score(lm_elastic, X[selected_features_rfe], y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
# cv_scores_elasticNet : array([-13.44161491, -12.67828467, -15.41808165, -15.1937048 , -15.53008139])
# --> No more Overfitting

# Tye to fit on log of time
lm_elastic.fit(X[selected_features_rfe], y.apply(lambda x: np.log(x+1)))

y_pred_lm_elastic = lm_elastic.predict(X_test[selected_features_rfe])
y_pred_lm_elastic = np.exp(y_pred_lm_elastic)-1
print('ElasticNet with selected_features_rfe on log(time_to_failure + 1)',
      mean_absolute_error(y_test, y_pred_lm_elastic), mean_squared_error(y_test, y_pred_lm_elastic))

#######################################
######    #######
#    #    #
# ####    ###
#   ##    #

# Grid Search RandomForestRegressor on 50 most important features
# rf_params = {'n_estimators': [10, 30, 100],
#              'criterion': ['mae'],
#              'max_depth': [5, 7],
#              'n_jobs': [-1],
#              'min_impurity_decrease': [0, 0.1]}
#
#
# grid_search_rf = GridSearchCV(RandomForestRegressor(),
#                               rf_params,
#                               scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'],
#                               refit='neg_mean_absolute_error',
#                               cv=5,
#                               verbose=2,
#                               n_jobs=-1)

# grid_search_rf.fit(X[selected_features_rf_50], y)
#
# grid_searchCV_results_rf = grid_search_rf.cv_results_
#
# grid_searchCV_results_rf = pd.DataFrame(grid_searchCV_results_rf)
# grid_searchCV_results_rf.to_csv("grid_searchCV_results_rf.csv")


# Using best parameters for MAE and Overfitting:
# rf = RandomForestRegressor(n_estimators=30, criterion='mae',
#                            max_depth=7, n_jobs=-1, min_impurity_decrease=0, verbose=2)
# rf.fit(X[selected_features_rf_50], y)
#
# X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
# y_pred_rf = rf.predict(X_test[selected_features_rf_50])
# pd.DataFrame(y_pred_rf).to_csv("y_pred_rf.csv")
y_pred_rf = pd.read_csv("y_pred_rf.csv")['0']

print('RandomForest Regressor with 50 Best features',
      mean_absolute_error(y_test, y_pred_rf), mean_squared_error(y_test, y_pred_rf))

plot_error_repartition(y_test, y_pred_rf, model_name='RandomForest_30', save=True)

# Train SVM on Best 50 features:

#####  #        #  ######
#       #      #   #     #
####     #    #    ######
   #      #  #     #    ##
####       ##      #    ##
# svr_params = {'kernel': ['sigmoid', 'poly'], 'gamma': ['scale'], 'C': [0.5, 1, 1.5]}
# grid_search_svr = GridSearchCV(SVR(),
#                                svr_params,
#                                scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'],
#                                refit='neg_mean_absolute_error',
#                                cv=5,
#                                verbose=2,
#                                n_jobs=-1)
#
# grid_search_svr.fit(X[selected_features_rf_50], y)
# grid_searchCV_results_svr = grid_search_svr.cv_results_
# pd.DataFrame(grid_searchCV_results_svr).to_csv("grid_searchCV_results_svr.csv")
# grid_searchCV_results_svr = pd.read_csv("grid_searchCV_results_svr.csv")
#
# svr = SVR(kernel='poly', gamma='scale', C=1.5)
# svr.fit(X[selected_features_rf_50], y)
# y_pred_svr = svr.predict(X_test[selected_features_rf_50])
# pd.DataFrame(y_pred_svr).to_csv('y_pred_svr.csv')
y_pred_svr = pd.read_csv('y_pred_svr.csv')['0']
print('SVR with 50 best feature', mean_absolute_error(y_test, y_pred_svr), mean_squared_error(y_test, y_pred_svr))
plot_error_repartition(y_test, y_pred_svr, model_name='SVR', save=True)


##      ##    ######
  ##  ##     #
    ##       #  ####
  ##  ##     #    ##
##      ##    ####


# def huber_loss_train(y_true, y_pred):
#     d = y_true - y_pred
#     h = 1  #h is delta in the graphic
#     scale = 1 + (d / h) ** 2
#     scale_sqrt = np.sqrt(scale)
#     grad = d / scale_sqrt
#     hess = 1 / scale / scale_sqrt
#     return grad, hess
#
#
# # define a loss that is MSE if time to failure is small (<50), and MAE otherwise.
# def custom_loss_train(y_true, y_pred):
#     grad = np.where(y_true < 50, huber_loss_train(y_true, y_pred)[0], 2*(y_true - y_pred))
#     hess = np.where(y_true < 50, huber_loss_train(y_true, y_pred)[1], 2)
#     return grad, hess


sample_weights = np.where(
    y <= 10, 100*10, np.where(
        y <= 20, 25*5, np.where(
            y <= 30, 10*4, np.where(
                y <= 40, 7*3, np.where(
                    y <= 50, 2, np.where(
                        y <= 100, 1, 0.1
                    ))))))

xg_params = {'max_depth': 3, 'objective': 'reg:squarederror'}
xgb_reg_weighted = xgb.XGBRegressor(**xg_params)
xgb_reg_weighted.fit(X[selected_features_rf_50], y, verbose=2, sample_weight=sample_weights)

xgb_reg = xgb.XGBRegressor(**xg_params)
xgb_reg.fit(X[selected_features_rf_50], y, verbose=2)


# evals_result = xgb_reg.evals_result()
# xgb_reg.save_model('xgb1.model')

xgb.plot_importance(xgb_reg)
plt.show()

y_pred_xgb = xgb_reg.predict(X_test[selected_features_rf_50])
y_pred_xgb_weighted = xgb_reg_weighted.predict(X_test[selected_features_rf_50])

print('XGB', mean_absolute_error(y_test, y_pred_xgb),
      mean_squared_error(y_test, y_pred_xgb))
print('XGB weighted', mean_absolute_error(y_test, y_pred_xgb),
      mean_squared_error(y_test, y_pred_xgb))

plot_error_repartition(y_test, y_pred_xgb, model_name='XGB', save=True)
plot_error_repartition(y_test, y_pred_xgb_weighted, model_name='XGB weighted', save=True)


residual_dual_plot(y_test, y_pred_xgb_weighted)
