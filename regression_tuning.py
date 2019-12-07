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
from sklearn.preprocessing import MinMaxScaler
from cross_validation import split_engines_for_cv


df_train = pd.read_csv('feature_engineered_train.csv').set_index('id')
df_test = pd.read_csv('feature_engineered_test.csv').set_index('id')

# df_train_not_capped = pd.read_csv('feature_engineered_train_not_capped.csv').set_index('id')
# df_test_not_capped = pd.read_csv('feature_engineered_test_not_capped.csv').set_index('id')

X, y = df_train.drop('time_to_failure', axis=1), df_train.time_to_failure
X_test, y_test = df_test.drop('time_to_failure', axis=1), df_test.time_to_failure



#
# # Recursive feature elimination
# rfe = RFECV(LinearRegression(), scoring='neg_mean_absolute_error', cv=5)
# rfe.fit(X, y)
# selected_features_rfe = [X.columns[i] for i in range(X.shape[1]) if rfe.get_support()[i]]
# print('selected RFECV features:', selected_features_rfe)
# pd.DataFrame(selected_features_rfe).to_csv("regression_results_and_plots/selected_features_rfe.csv")

selected_features_rfe = pd.read_csv("regression_results_and_plots/selected_features_rfe.csv")['0']
selected_features_rfe = selected_features_rfe.values

removed_features = list(X.columns)
for f in selected_features_rfe:
    removed_features.remove(f)

# removed_features Out[46]: ['sensor9_time_reversal_asymmetry_window_20_lag_5',
# 'sensor14_time_reversal_asymmetry_window_20_lag_5']
# --> Only removes 2 features out of 169 !
# Looking closer, it turns out there is a scale problem with these 2

scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
# after performing a RFE on the scaled features, every features are kept.
selected_features_scaled_rfe = pd.read_csv("regression_results_and_plots/selected_scaled_features_rfe.csv")['0']
selected_features_scaled_rfe = selected_features_scaled_rfe.values

# --> We do not get rid of any feature yet.

####
####
# We now want to look at feature importance.
# For this, we use a Tree based method to better leverage correlation of our features.

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
# rf.fit(X_scaled, y)
# toc = datetime.datetime.now()
# print(toc-tic)
# feat_imp = pd.DataFrame(X.columns, columns=['feat_name'])
# feat_imp['imp'] = rf.feature_importances_
# feat_imp.to_csv("regression_results_and_plots/rf_feat_scaled_importance.csv")
feat_imp = pd.read_csv("regression_results_and_plots/rf_feat_scaled_importance.csv")
feat_imp.sort_values(by='imp', ascending=False, inplace=True)

plt.plot(feat_imp.imp.values.cumsum())
plt.grid()
plt.savefig("regression_results_and_plots/cumulated scaled-feature importance random forest")
plt.show()

# 25 features keep 90% of the feature importance
# 50 features keep 97% of the feature importance
# 75 features keep 99% of the feature importance

plt.plot(feat_imp.imp.values[25:].cumsum())
plt.grid()
plt.show()

selected_features_rf_50 = feat_imp.feat_name.values[:50]
selected_features_rf_25 = feat_imp.feat_name.values[:25]
# selected_features_rf_25:
# 'sensor11_diff_lag_20', 'sensor11_rolling_mean_derivative_20',
#        'sensor2_rolling_max_20', 'sensor11_rolling_sum_of_changes_20',
#        'sensor2', 'sensor9_diff_lag_20',
#        'sensor14_rolling_sum_of_changes_20', 'sensor9_delta_lag_20',
#        'sensor13_rolling_mean_derivative_20', 'sensor20_log',
#        'sensor4_rolling_max_20', 'sensor13_diff_lag_20', 'sensor17_log',
#        'sensor2_log', 'sensor4_log', 'sensor21_log', 'sensor7_log',
#        'sensor17', 'sensor4', 'sensor9_rolling_mean_derivative_20',
#        'sensor4_rolling_mean_derivative_20', 'sensor7_rolling_min_20',
#        'sensor12_rolling_sum_of_changes_20', 'sensor3', 'sensor20'],
#       dtype=object)
#
#############
#######################################################################################################
# DONE WITH FEATURE SELECTION --> GOING TO MODEL SELECTION ###################################
#######################################################################################################
############
#

#       #     #
#       ##   ##
#       # # # #
#####   #  #  #

lin_reg = LinearRegression()
lin_reg.fit(X_scaled, y)  # No possibility to fit on MAE. Only OLS.
y_pred_lm = lin_reg.predict(X_test_scaled)
print('Linear Regression',
      mean_absolute_error(y_test, y_pred_lm), mean_squared_error(y_test, y_pred_lm))
# Linear Regression 14.186551385453248 320.8556628639239

cross_val_score(lin_reg, X_scaled, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
# array([-12.16980952, -11.17104961, -12.59957027, -12.58221051, -12.94705154])
cross_val_score(lin_reg, X_scaled, y, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
# array([-240.76934135, -199.44898308, -268.04097642, -273.95922803, -274.64024234])
residual_quadra_plot(np.array(y_test), np.array(y_pred_lm), model_name='Linear Regression', save=False)

# lin_reg.fit(X_scaled[selected_features_rf_50], y)
y_pred_lm_50 = lin_reg.predict(X_test_scaled[selected_features_rf_50])
print('Linear Regression',
      mean_absolute_error(y_test, y_pred_lm_50), mean_squared_error(y_test, y_pred_lm_50))
# Linear Regression 15.62867085192837 386.20531142859625
lin_reg = LinearRegression()
cross_val_score(lin_reg, X_scaled[selected_features_rf_50], y, scoring='neg_mean_absolute_error', cv=2, n_jobs=-1)
# array([-13.74193944, -15.29926381])
# Still overfitt
cross_val_score(lin_reg, X_scaled[selected_features_rf_50], y, scoring='neg_mean_absolute_error',
                cv=split_engines_for_cv(train_df, 5), n_jobs=-1)



lin_reg.fit(X_scaled[selected_features_rf_25], y)
y_pred_lm_25 = lin_reg.predict(X_test_scaled[selected_features_rf_25])
print('Linear Regression',
      mean_absolute_error(y_test, y_pred_lm_25), mean_squared_error(y_test, y_pred_lm_25))
# 16.510952094718043 423.09788009478444
cross_val_score(lin_reg, X_scaled[selected_features_rf_25], y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
# array([-14.6766886 , -16.33929278])

#####  #   #
##     ##  #
#####  # # #
##     #  ##
#####  #   #

# Train ElasticNet regression:
# l1_ratio = [0.1, 0.3, 0.7, 0.9, 0.99, 1]
# normalize = [True, False]
# n_jobs = -1
# max_iter = 100000
# cv = 5
# lm_elasticCV = ElasticNetCV(l1_ratio=l1_ratio,
#                           normalize=normalize,
#                           n_jobs=n_jobs,
#                           max_iter=max_iter,
#                           cv=cv,
#                           verbose=1)
# lm_elasticCV.fit(X_scaled, y)  # Only MSE Loss available.
# lm_elasticCV.alpha_
# 0.000311421461336888
#
# lm_elasticCV.l1_ratio_
# 1.0
# y_pred_lm_elasticCV = lm_elasticCV.predict(X_test_scaled)
# print('ElasticNetCV',
#       mean_absolute_error(y_test, y_pred_lm_elasticCV), mean_squared_error(y_test, y_pred_lm_elasticCV))
# 16.789624747721206 441.04810419595884
#
# lm_elastic_params = {'alpha': 0.00031142146133688865,
#  'l1_ratio': 1,
#  'max_iter': 50000}
#
# lm_elastic = ElasticNet().set_params(**lm_elastic_params)
# lm_elastic.fit(X_scaled, y)    # Only MSE Loss available.
#
# y_pred_lm_elastic = lm_elastic.predict(X_test_scaled)
# pd.DataFrame(y_pred_lm_elastic).to_csv("regression_results_and_plots/y_pred_lm_elastic.csv")
y_pred_lm_elastic = pd.read_csv("regression_results_and_plots/y_pred_lm_elastic.csv")['0']
print('ElasticNet',
      mean_absolute_error(y_test, y_pred_lm_elastic), mean_squared_error(y_test, y_pred_lm_elastic))
# ElasticNet 16.143727696448916 420.01787048073106
residual_quadra_plot(np.array(y_test), np.array(y_pred_lm_elastic), model_name='ElasticNet', save=False)


# Tye to fit on log of time
# lm_elastic_log = ElasticNet().set_params(**lm_elastic_params)
# lm_elastic_log.fit(X_scaled, y.apply(lambda x: np.log(x+1)))
#
# y_pred_lm_elastic_log = lm_elastic_log.predict(X_test_scaled)
# y_pred_lm_elastic_log = np.exp(y_pred_lm_elastic_log)-1
# pd.DataFrame(y_pred_lm_elastic_log).to_csv("regression_results_and_plots/y_pred_lm_elastic_log.csv")
y_pred_lm_elastic_log = pd.read_csv("regression_results_and_plots/y_pred_lm_elastic_log.csv")['0']

print('ElasticNet with selected_features_rfe on log(time_to_failure + 1)',
      mean_absolute_error(y_test, y_pred_lm_elastic_log), mean_squared_error(y_test, y_pred_lm_elastic_log))
# 18.58649892751042 574.5034165674568
residual_quadra_plot(np.array(y_test), np.array(y_pred_lm_elastic_log), model_name='ElasticNet on log(RUL+1)')


#######################################
######    #######
#    #    #
# ####    ###
#   ##    #

# Extremely slow

# Grid Search RandomForestRegressor on 50 most important features
# rf_params = {'n_estimators': [30],
#              'criterion': ['mae'],
#              'max_depth': [4, 7],
#              'n_jobs': [-1],
#              'min_impurity_decrease': [0]}
#
#
# grid_search_rf = GridSearchCV(RandomForestRegressor(),
#                               rf_params,
#                               scoring='neg_mean_absolute_error',
#                               cv=3,
#                               verbose=2,
#                               n_jobs=-1)
#
# grid_search_rf.fit(X_scaled[selected_features_rf_50], y)
#
# grid_searchCV_results_rf = grid_search_rf.cv_results_
#
# grid_searchCV_results_rf = pd.DataFrame(grid_searchCV_results_rf)
# grid_searchCV_results_rf.to_csv("regression_results_and_plots/grid_searchCV_results_rf_3.csv")

grid_searchCV_results_rf = pd.read_csv("regression_results_and_plots/grid_searchCV_results_rf_3.csv")
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
rf = RandomForestRegressor(n_estimators=30, criterion='mae',
                           max_depth=7, n_jobs=-1, min_impurity_decrease=0, verbose=2)
rf.fit(X_scaled[selected_features_rf_50], y)

y_pred_rf = rf.predict(X_test_scaled[selected_features_rf_50])
pd.DataFrame(y_pred_rf).to_csv("regression_results_and_plots/y_pred_rf_30_7.csv")
y_pred_rf = pd.read_csv("regression_results_and_plots/y_pred_rf_30_7.csv")['0']

print('RandomForest Regressor with 50 Best features',
      mean_absolute_error(y_test, y_pred_rf), mean_squared_error(y_test, y_pred_rf))
# 50 best features, (n_estimators=30, max_depth=7) 14.278511610897787 396.8144270128604

cv_scores_lin_reg = cross_val_score(rf, X_scaled, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)


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
    grad = np.where(y_true < 50, (huber_approx_obj(y_pred, y_true)[0])**2, huber_approx_obj(y_pred, y_true)[0])
    hess = np.where(y_true < 50, huber_approx_obj(y_pred, y_true)[1]*2, huber_approx_obj(y_pred, y_true)[1])
    return grad, hess

def custom_loss_treshold_50(y_pred, y_true):
    grad = np.where(y_true < 50, 2.5*(y_true-y_pred), 1)
    hess = np.where(y_true < 50, 2.5, 0)
    return grad, hess

def custom_loss_treshold_75(y_pred, y_true):
    grad = np.where(y_true < 75, 2.5*(y_true-y_pred), 1)
    hess = np.where(y_true < 75, 2.5, 0)
    return grad, hess

def custom_loss(y_pred, y_true):
    grad = np.where((y_true-y_pred) < 75, 2.5*(y_true-y_pred), 1)
    hess = np.where((y_true-y_pred) < 75, 2.5, 0)
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

for obj in [custom_loss_treshold_50, custom_loss_treshold_75, custom_loss, 'reg:gamma', 'reg:tweedie', 'reg:squarederror']:
    if obj == huber_approx_obj:
        XGB = xgb.XGBRegressor(max_depth=3, objective=obj, n_estimators=200, n_jobs=-1)
    else:
        XGB = xgb.XGBRegressor(max_depth=3, objective=obj, n_estimators=100, n_jobs=-1)
    XGB.fit(X, y, verbose=2)
    y_pred = XGB.predict(X_test)
    if callable(obj):
        obj_name = obj.__name__
    else:
        obj_name = obj
    y_pred_xgb[obj_name] = y_pred
    print("XGB with {objective}".format(objective=obj),
          mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred))
    residual_quadra_plot(y_test, y_pred, model_name="XGB with {obj_name}".format(obj_name=obj_name), save=True)
y_pred_xgb.to_csv("regression_results_and_plots/y_pred_xgb_5_obj.csv")

# XGB with custom_loss_treshold_50 30.93 1680
# XGB with custom_loss_treshold_75 29.60 1435
# XGB with custom_loss             12.48 257
# XGB with reg:gamma               13.64 292
# XGB with reg:tweedie             12.25 253
# XGB with reg:squarederror        12.44 256

#
sample_weights = np.where(
    y <= 10, 100*10, np.where(
        y <= 20, 25*5, np.where(
            y <= 30, 10*4, np.where(
                y <= 40, 7*3, np.where(
                    y <= 50, 2, np.where(
                        y <= 100, 1, 0.1
                    ))))))

# xgb_reg_weighted_tweedie = xgb.XGBRegressor(max_depth=3, objective='reg:tweedie', n_estimators=200, n_jobs=-1)
# xgb_reg_weighted_tweedie.fit(X, y, verbose=2, sample_weight=sample_weights)
# y_pred_weighted_tweedie = xgb_reg_weighted_tweedie.predict(X_test)
# pd.DataFrame(y_pred_weighted_tweedie).to_csv("regression_results_and_plots/y_pred_weighted_tweedie.csv")
y_pred_weighted_tweedie = np.array(pd.read_csv("regression_results_and_plots/y_pred_weighted_tweedie.csv")['0'])
print("XGB weighted tweedie", mean_absolute_error(y_test, y_pred_weighted_tweedie), mean_squared_error(y_test, y_pred_weighted_tweedie))
# XGB weighted tweedie 30.674529933824818 1384.2559447988936
residual_quadra_plot(y_test, y_pred_weighted_tweedie, model_name="XGB weighted tweedie", save=True)

xgb_reg_weighted_squarederror = xgb.XGBRegressor(max_depth=3, objective='reg:squarederror', n_estimators=200, n_jobs=-1)
xgb_reg_weighted_squarederror.fit(X, y, verbose=2, sample_weight=sample_weights)
y_pred_weighted_squarederror = xgb_reg_weighted_squarederror.predict(X_test)
pd.DataFrame(y_pred_weighted_squarederror).to_csv("regression_results_and_plots/y_pred_weighted_squarederror.csv")
y_pred_weighted_squarederror = np.array(pd.read_csv("regression_results_and_plots/y_pred_weighted_squarederror.csv")['0'])
print("XGB weighted squarederror", mean_absolute_error(y_test, y_pred_weighted_squarederror), mean_squared_error(y_test, y_pred_weighted_squarederror))
# XGB weighted sqarederror 16.108829582672623 436.2950446098589
residual_quadra_plot(y_test, y_pred_weighted_squarederror, model_name="XGB weighted squarederror", save=True)

#
# cv_scores_xgb = cross_val_score(
#     xgb_reg_weighted, X, y, scoring='neg_mean_absolute_error', cv=2, n_jobs=-1, verbose=2)
# # array([ -9.93397797, -12.24655596])
#
# # With 5 folds:
# cv_scores_xgb = cross_val_score(
#     xgb_reg_weighted, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=2)
# # cv_scores_xgb_50_features = array([ -9.04913295,  -9.29853543,  -9.3275443 , -11.59707833, -11.53495859])
#
# xgb_reg_weighted_squarederror_feature_importances = pd.DataFrame(xgb_reg_weighted_squarederror.feature_importances_)
# xgb_reg_weighted_squarederror_feature_importances.to_csv("xgb_reg_weighted_squarederror_feature_importances.csv")
xgb_reg_weighted_squarederror_feature_importances = pd.read_csv("xgb_reg_weighted_squarederror_feature_importances.csv")['0']
plt.plot(-np.sort(-xgb_reg_weighted_squarederror.feature_importances_).cumsum())
plt.title('feature importance for xgb_reg_weighted_squarederror')
# plt.savefig("feature importance for xgb_reg_weighted_squarederror")
plt.show()
# --> Let's keep the 100 most important features for XG boost, who account for more than 99% of the importance.
selected_features_xgb_100 = pd.DataFrame(data=X.columns.values, columns=['feat_names'])
selected_features_xgb_100['feat_imp'] = xgb_reg_weighted_squarederror_feature_importances
selected_features_xgb_100.sort_values(by='feat_imp')
selected_features_xgb_100 = selected_features_xgb_100.feat_names[:100]
selected_features_xgb_75 = selected_features_xgb_100[:75]
selected_features_xgb_50 = selected_features_xgb_100[:50]

cv_scores_xgb = cross_val_score(
    xgb_reg_weighted_squarederror, X[selected_features_xgb_100], y, scoring='neg_mean_absolute_error', cv=2, n_jobs=-1, verbose=2)
# [ -9.58242875,  -9.80860827,  -9.38500866, -12.01706412, -11.58635128])
# array([-10.62010572, -12.91100659])

cv_scores_xgb = cross_val_score(
    xgb_reg_weighted_squarederror, X[selected_features_xgb_75], y, scoring='neg_mean_absolute_error', cv=2, n_jobs=-1, verbose=2)
# Out[31]: array([-10.61753812, -13.00911618])

cv_scores_xgb = cross_val_score(
    xgb_reg_weighted_squarederror, X[selected_features_xgb_50], y, scoring='neg_mean_absolute_error', cv=2, n_jobs=-1, verbose=2)
#Out[33]: array([-10.53843629, -12.67522863])