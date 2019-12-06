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
# rfe = RFECV(LinearRegression(), scoring='neg_mean_absolute_error', cv=5)
# rfe.fit(X, y)
# selected_features_rfe = [X.columns[i] for i in range(X.shape[1]) if rfe.get_support()[i]]
# print('selected RFECV features:', selected_features_rfe)
# pd.DataFrame(selected_features_rfe).to_csv("regression_results_and_plots/selected_features_rfe.csv")

selected_features_rfe = pd.read_csv("regression_results_and_plots/selected_features_rfe.csv")['0']
selected_features_rfe = selected_features_rfe.values

# --> Keeps 167 features out of 170.  Not very helpfull.
# Also, do not account for interaction between features as uses linear regression.


# Select k most relevant features
# Play around with K to see which number of features is best
# Statistical Analysis of features.
# kbest = SelectKBest(k=100)
# kbest.fit(X, y)
# selected_features_kbest = [X.columns[i] for i in range(X.shape[1]) if kbest.get_support()[i]==True]
# print('selected kbest features:', selected_features_kbest)


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
feat_imp = pd.read_csv("regression_results_and_plots/rf_feat_importance.csv")
feat_imp.sort_values(by='imp', ascending=False, inplace=True)

plt.plot(feat_imp.imp.values.cumsum())
plt.grid()
# plt.savefig("regression_results_and_plots/cumulated feature importance random forest")
plt.show()

plt.plot(feat_imp.imp.values[25:].cumsum())
plt.grid()
plt.show()

selected_features_rf_50 = feat_imp.feat_name.values[:50]
selected_features_rf_25 = feat_imp.feat_name.values[:25]

#
#############
#######################################################################################################
# DONE WITH FEATURE SELECTION --> GOING TO MODEL SELECTION ###################################
#######################################################################################################
############
#

# lin_reg = LinearRegression()
# lin_reg.fit(X, y)
#
# y_pred_lm = lin_reg.predict(X_test)
# print('Linear Regression',
#       mean_absolute_error(y_test, y_pred_lm), mean_squared_error(y_test, y_pred_lm))
# # Linear Regression 37.2388022132974 2286.8191973408284
#
#
# # Train LinearRegression on RFE features
# lin_reg = LinearRegression()
# lin_reg.fit(X[selected_features_rfe], y)
# y_pred_lm = lin_reg.predict(X_test[selected_features_rfe])
# pd.DataFrame(y_pred_lm).to_csv("regression_results_and_plots/y_pred_lm.csv")
y_pred_lm = pd.read_csv("regression_results_and_plots/y_pred_lm.csv")['0']

print('Linear Regression with RFE feature selection',
      mean_absolute_error(y_test, y_pred_lm), mean_squared_error(y_test, y_pred_lm))
# Linear Regression with RFE feature selection 14.142734642572869 319.7344583322316


# cv_scores_lin_reg = cross_val_score(lin_reg, X[selected_features_rfe], y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
# cv_scores_lin_reg = array([-12.17061674, -11.10523918, -12.81084418, -12.70238884, -12.9531474 ])
# --> Overfitting

# plot_error_repartition(y_test, y_pred_lm, model_name='Linear Regression', save=True)
residual_quadra_plot(np.array(y_test), np.array(y_pred_lm), model_name='Linear Regression', save=False)

#####  #   #
##     ##  #
#####  # # #
##     #  ##
#####  #   #

# Train ElasticNet regression:
# l1_ratio = [0.1, 0.3, 0.7, 0.9, 0.99, 1]
# normalize = [True, False]
# n_jobs = 3
# max_iter = 100000
# cv = 5
# lm_elasticCV = ElasticNetCV(l1_ratio=l1_ratio,
#                           normalize=normalize,
#                           n_jobs=n_jobs,
#                           max_iter=max_iter,
#                           cv=cv,
#                           verbose=2)
# lm_elasticCV.fit(X, y)
# alpha = lm_elasticCV.alpha_
# # 0.00031142146133688865
#
# l1_ratio = lm_elasticCV.l1_ratio_
# 1.0
# y_pred_lm_elasticCV = lm_elasticCV.predict(X_test)
# print('ElasticNet with selected_features_rfe',
#       mean_absolute_error(y_test, y_pred_lm_elasticCV), mean_squared_error(y_test, y_pred_lm_elasticCV))
# 16.789624747721206 441.04810419595884
#
# lm_elastic_params = {'alpha': 0.00031142146133688865,
#  'l1_ratio': 1,
#  'max_iter': 50000}
#
# lm_elastic = ElasticNet().set_params(**lm_elastic_params)
# lm_elastic.fit(X, y)
#
# y_pred_lm_elastic = lm_elastic.predict(X_test)
# pd.DataFrame(y_pred_lm_elastic).to_csv("regression_results_and_plots/y_pred_lm_elastic.csv")
y_pred_lm_elastic = pd.read_csv("regression_results_and_plots/y_pred_lm_elastic.csv")['0']
print('ElasticNet',
      mean_absolute_error(y_test, y_pred_lm_elastic), mean_squared_error(y_test, y_pred_lm_elastic))
# ElasticNet 16.411964629811976 430.32060553752626
residual_quadra_plot(np.array(y_test), np.array(y_pred_lm_elastic), model_name='ElasticNet', save=False)


# lm_elastic_rfe.fit(X[selected_features_rfe], y)
# y_pred_lm_elastic_rfe = lm_elastic_rfe.predict(X_test[selected_features_rfe])
# print('ElasticNet with selected_features_rfe',
#       mean_absolute_error(y_test, y_pred_lm_elastic_rfe), mean_squared_error(y_test, y_pred_lm_elastic_rfe))
# ElasticNet with selected_features_rfe 16.413186887370703 432.07932720282804

# plot_error_repartition(y_test, y_pred_lm_elastic, model_name='ElasticNet', save=True)
# residual_quadra_plot(y_test, y_pred_lm_elastic)
# cv_scores_elasticNet = cross_val_score(lm_elastic, X[selected_features_rfe], y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
# cv_scores_elasticNet : array([-13.44161491, -12.67828467, -15.41808165, -15.1937048 , -15.53008139])
# --> No more Overfitting

# Tye to fit on log of time
# lm_elastic_log = ElasticNet().set_params(**lm_elastic_params)
# lm_elastic_log.fit(X, y.apply(lambda x: np.log(x+1)))

# y_pred_lm_elastic_log = lm_elastic_log.predict(X_test)
# y_pred_lm_elastic_log = np.exp(y_pred_lm_elastic_log)-1
# pd.DataFrame(y_pred_lm_elastic_log).to_csv("regression_results_and_plots/y_pred_lm_elastic_log.csv")
y_pred_lm_elastic_log = pd.read_csv("regression_results_and_plots/y_pred_lm_elastic_log.csv")['0']

print('ElasticNet with selected_features_rfe on log(time_to_failure + 1)',
      mean_absolute_error(y_test, y_pred_lm_elastic_log), mean_squared_error(y_test, y_pred_lm_elastic_log))
# 17.235873705938257 505.43806961537985
residual_quadra_plot(np.array(y_test), np.array(y_pred_lm_elastic_log), model_name='ElasticNet on log(RUL+1)', save=False)


#######################################
######    #######
#    #    #
# ####    ###
#   ##    #

# Extremely slow

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

grid_searchCV_results_rf = pd.read_csv("regression_results_and_plots/grid_searchCV_results_rf.csv")
grid_searchCV_results_rf = pd.DataFrame(grid_searchCV_results_rf)
grid_searchCV_results_rf


cols = ['param_criterion', 'param_max_depth',
       'param_min_impurity_decrease', 'param_n_estimators',
       'mean_test_neg_mean_absolute_error', 'std_test_neg_mean_absolute_error',
       'rank_test_neg_mean_absolute_error',
       'rank_test_neg_mean_squared_error']
new_cols = ['criterion', 'max depth',
       'min impurity decrease', 'n estimators',
       'mean mae', 'std mae',
       'rank mae',
       'rank mse']
grid_searchCV_results_rf = grid_searchCV_results_rf[cols]
grid_searchCV_results_rf.columns = new_cols
grid_searchCV_results_rf[['mean mae', 'std mae']] = grid_searchCV_results_rf[['mean mae', 'std mae']].round(2)

render_mpl_table(grid_searchCV_results_rf, header_columns=0, col_width=2.1)
# plt.savefig("regression_results_and_plots/grid_searchCV_results_rf")
plt.show()

# Using best parameters for MAE and Overfitting:
# rf = RandomForestRegressor(n_estimators=30, criterion='mae',
#                            max_depth=7, n_jobs=-1, min_impurity_decrease=0, verbose=2)
# rf.fit(X[selected_features_rf_50], y)
#
# X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure
# y_pred_rf = rf.predict(X_test[selected_features_rf_50])
# pd.DataFrame(y_pred_rf).to_csv("y_pred_rf.csv")
y_pred_rf = pd.read_csv("regression_results_and_plots/y_pred_rf.csv")['0']

print('RandomForest Regressor with 50 Best features',
      mean_absolute_error(y_test, y_pred_rf), mean_squared_error(y_test, y_pred_rf))
# 14.278511610897787 396.8144270128604

plot_error_repartition(y_test, y_pred_rf, model_name='RandomForest', save=False)
residual_quadra_plot(np.array(y_test), np.array(y_pred_rf), model_name="Random Forest", save=False)


#####  #        #  ######
#       #      #   #     #
####     #    #    ######
   #      #  #     #    ##
####       ##      #    ##


# svr_params = {'kernel': ['sigmoid', 'poly', 'rbf']}
# grid_search_svr = GridSearchCV(SVR(),
#                                svr_params,
#                                scoring=['neg_mean_absolute_error'],
#                                refit='neg_mean_absolute_error',
#                                cv=5,
#                                verbose=2,
#                                n_jobs=-1)
#
# grid_search_svr.fit(X[selected_features_rf_50], y)
# grid_searchCV_results_svr = grid_search_svr.cv_results_
# pd.DataFrame(grid_searchCV_results_svr).to_csv("regression_results_and_plots/grid_searchCV_results_svr.csv")
# grid_searchCV_results_svr = pd.read_csv("regression_results_and_plots/grid_searchCV_results_svr.csv")
#
# best_svr_params = grid_search_svr.best_params_
# best_svr_params = {'kernel': 'rbf'}
# svr = SVR(**best_svr_params)
# svr.fit(X[selected_features_rf_50], y)
# y_pred_svr = svr.predict(X_test[selected_features_rf_50])
# pd.DataFrame(y_pred_svr).to_csv('regression_results_and_plots/y_pred_svr.csv')
y_pred_svr = pd.read_csv('regression_results_and_plots/y_pred_svr.csv')['0']
print('SVR with 50 best feature', mean_absolute_error(y_test, y_pred_svr), mean_squared_error(y_test, y_pred_svr))
# 19.71195007675904 711.9330696763042
# plot_error_repartition(y_test, y_pred_svr, model_name='SVR', save=True)
residual_quadra_plot(np.array(y_test), np.array(y_pred_svr), model_name="SVR", save=True)

##      ##    ######
  ##  ##     #
    ##       #  ####
  ##  ##     #    ##
##      ##    ####


def huber_approx_obj(preds, dtrain):
    d = preds - dtrain
    h = 1
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = -d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


# define a loss that is MSE if time to failure is small (<50), and MAE otherwise.
def custom_loss_train(y_pred, y_true):
    grad = np.where(y_true < 50, huber_approx_obj(y_pred, y_true)[0], (huber_approx_obj(y_pred, y_true)[0])**2)
    hess = np.where(y_true < 50, huber_approx_obj(y_pred, y_true)[1], (huber_approx_obj(y_pred, y_true)[1])*2)
    return grad, hess


# xg_params = {'max_depth': [3, 4, 7],
#              'objective': [huber_approx_obj, custom_loss_train, 'reg:gamma', 'reg:tweedie', 'reg:squarederror']
#              'n_estimators': [50, 100, 200]
#              }
# #  'objective': [huber_approx_obj, custom_loss_train, 'reg:gamma', 'reg:tweedie', 'reg:squarederror']
# XGB = xgb.XGBRegressor()
# grid_search_xgb = GridSearchCV(XGB,
#                                xg_params,
#                                scoring='neg_mean_absolute_error',
#                                cv=5,
#                                verbose=2,
#                                n_jobs=-1)
# grid_search_xgb.fit(X, y)
#
# pd.DataFrame(grid_search_xgb.cv_results_).to_csv("regression_results_and_plots/grid_search_xgb_results_custom_with_depth3_4.csv")
# grid_search_xgb_results = pd.read_csv("regression_results_and_plots/grid_search_xgb_results_custom_with_depth3_4.csv")

y_pred_xgb = pd.DataFrame()

for obj in [huber_approx_obj, custom_loss_train, 'reg:gamma', 'reg:tweedie', 'reg:squarederror']:
    if obj == huber_approx_obj:
        XGB = xgb.XGBRegressor(max_depth=3, objective=obj, n_estimators=200)
    else:
        XGB = xgb.XGBRegressor(max_depth=3, objective=obj, n_estimators=100)
    XGB.fit(X, y, verbose=2)
    y_pred = XGB.predict(X_test)
    if callable(obj):
        obj_name = obj.__name__
    else:
        obj_name = obj
    y_pred_xgb[obj_name] = y_pred
    print("XGB with {objective}".format(objective=obj),
          mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred))
    residual_quadra_plot(y_test, y_pred, model_name="XGB with {objective}".format(objective=obj), save=False)

y_pred_xgb.to_csv("regression_results_and_plots/y_pred_xgb_5_obj.csv")

xgb_reg_all_features = xgb.XGBRegressor(**xg_params)
xgb_reg_all_features.fit(X, y, verbose=2, eval_set=[(X, y), (X_test, y_test)], eval_metric='logloss')
y_pred_xgb_all_features = xgb_reg_all_features.predict(X_test)
print('XGB all features', mean_absolute_error(y_test, y_pred_xgb_all_features),
      mean_squared_error(y_test, y_pred_xgb_all_features))



feat_imp_xgb = pd.DataFrame(X.columns, columns=['feat_name'])
feat_imp_xgb['imp'] = xgb_reg_50.feature_importances_
selected_features_xgb_35 = feat_imp_xgb.sort_values(by='imp')


sample_weights = np.where(
    y <= 10, 100*10, np.where(
        y <= 20, 25*5, np.where(
            y <= 30, 10*4, np.where(
                y <= 40, 7*3, np.where(
                    y <= 50, 2, np.where(
                        y <= 100, 1, 0.1
                    ))))))



xgb_reg_50 = xgb.XGBRegressor(**xg_params)
xgb_reg_50.fit(X[selected_features_rf_50], y, verbose=2)
y_pred_xgb = xgb_reg_50.predict(X_test[selected_features_rf_50])

print('XGB_50', mean_absolute_error(y_test, y_pred_xgb),
      mean_squared_error(y_test, y_pred_xgb))
# XGB_50 13.053960266992148 282.7311349215896
# For y_test.shape =(9115,)

# Let's make sur we do not overfitt: X.shape: (16631, 169)
# --> We make a 2 fold cv and should find something around 13 MAE

cv_scores_xgb_50_features = cross_val_score(
    xgb_reg_50, X[selected_features_rf_50], y, scoring='neg_mean_absolute_error', cv=2, n_jobs=-1)
# array([-10.67466953, -13.01657841])

# With 5 folds:
cv_scores_xgb_50_features = cross_val_score(
    xgb_reg_50, X[selected_features_rf_50], y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
# cv_scores_xgb_50_features = array([-10.17656852, -10.14356455,  -9.963878  , -12.32954861, -11.71705757])

plt.plot(-np.sort(-xgb_reg_50.feature_importances_).cumsum())
plt.show()
# --> Let's keep the 35 most important features for XG boost, who account for more than 99% of the importance.



xgb_reg_25 = xgb.XGBRegressor(**xg_params)
xgb_reg_25.fit(X[selected_features_rf_25], y, verbose=2)

xgb_reg_weighted = xgb.XGBRegressor(**xg_params)
xgb_reg_weighted.fit(X[selected_features_rf_50], y, verbose=2, sample_weight=sample_weights)



xgb_reg_all_features_weighted = xgb.XGBRegressor(**xg_params)
xgb_reg_all_features_weighted.fit(X, y, verbose=2, sample_weight=sample_weights)


cv_scores_xgb_all_features = cross_val_score(xgb_reg_all_features, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
# cv_scores_xgb_all_features = array([ -9.74721651, -10.13252724, -10.13221739, -12.32682284, -12.21885899])

xgb.plot_importance(xgb_reg_50)
plt.show()

y_pred_xgb_weighted = xgb_reg_weighted.predict(X_test[selected_features_rf_50])
y_pred_xgb_all_features_weighted = xgb_reg_all_features.predict(X_test)

print('XGB', mean_absolute_error(y_test, y_pred_xgb),
      mean_squared_error(y_test, y_pred_xgb))
print('XGB weighted', mean_absolute_error(y_test, y_pred_xgb),
      mean_squared_error(y_test, y_pred_xgb))


plot_error_repartition(y_test, y_pred_xgb, model_name='XGB', save=True)
plot_error_repartition(y_test, y_pred_xgb_weighted, model_name='XGB weighted', save=True)
plot_error_repartition(y_test, y_pred_xgb_all_features, model_name='XGB all features', save=True)

residual_quadra_plot(y_test, y_pred_xgb, model_name='XGB', save=True)
residual_quadra_plot(y_test, y_pred_xgb_weighted, model_name='XGB weighted', save=True)
residual_quadra_plot(y_test, y_pred_xgb_all_features, model_name='XGB all features', save=True)

