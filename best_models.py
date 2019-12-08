from sklearn.linear_model import LogisticRegressionCV

classification_best_model = {
    'estimator': LogisticRegressionCV,
    'feature_subset': ['sensor4', 'sensor8', 'sensor9', 'sensor11', 'sensor17', 'sensor17_rolling_mean_20',
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
}


sample_weights_2 = np.where(
    y <= 10, 10000, np.where(
        y <= 20, 1250, np.where(
            y <= 30, 150, np.where(
                y <= 40, 30, np.where(
                    y <= 50, 8, np.where(
                        y <= 100, 1, 0.1
                    ))))))

xgb_reg_weighted_squarederror = xgb.XGBRegressor(max_depth=3, objective='reg:squarederror', n_estimators=200, n_jobs=-1)
xgb_reg_weighted_squarederror.fit(X, y, verbose=2, sample_weight=sample_weights_2)
y_pred_weighted_squarederror = xgb_reg_weighted_squarederror.predict(X_test)
print("XGB weighted squarederror", mean_absolute_error(y_test, y_pred_weighted_squarederror), mean_squared_error(y_test, y_pred_weighted_squarederror))
plots.residual_quadra_plot(y_test, y_pred_weighted_squarederror, model_name="XGB weighted_2 squarederror", save=True)

