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

