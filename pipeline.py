import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plots
import feature_engineering
from preprocessing import generate_labels, denoise

from best_models import classification_best_model


# CMAPSS dataset number
dataset_number = 1

# Read Data
columns = ["id", "time", "op1", "op2", "op3"]
sensor_columns = ["sensor{}".format(i+1) for i in range(21)]
columns.extend(sensor_columns)
train_df = pd.read_csv("CMAPSSData/CMAPSSData/train_FD00{}.txt".format(dataset_number),
                       delimiter=" ", index_col=None, header=None)
test_df = pd.read_csv("CMAPSSData/CMAPSSData/test_FD00{}.txt".format(dataset_number),
                      delimiter=" ", index_col=None, header=None)

# Format data
train_df, test_df = train_df.dropna(axis=1), test_df.dropna(axis=1)
train_df.columns, test_df.columns = columns, columns
train_df.id, test_df.id = train_df.id.astype(np.int32), test_df.id.astype(np.int32)
train_df.time, test_df.time = train_df.time.astype(np.int32), test_df.time.astype(np.int32)
train_df, test_df = train_df.set_index('id'), test_df.set_index('id')

if dataset_number % 2 == 1:  # The data contains only one operating condition
    train_df = train_df.drop(["op1", "op2", "op3"], axis=1)
    test_df = test_df.drop(["op1", "op2", "op3"], axis=1)

label_file = "CMAPSSData/CMAPSSData/RUL_FD00{}.txt".format(dataset_number)
train_df, test_df = generate_labels(train_df, test_df, label_file)
print('Generated labels')

plots.plot_superposed_engine_signals(range(1, 101), train_df, sensor_columns)
filtered_signals = ['sensor{}'.format(i) for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
dropped_signals = ['sensor{}'.format(i) for i in (1, 5, 6, 10, 16, 18, 19)]

print('Dropping constant signals...')
train_df, test_df = train_df.drop(dropped_signals, axis=1), test_df.drop(dropped_signals, axis=1)
print('Filtered signals :', *filtered_signals)


plots.plot_denoising_process(train_df.loc[1]['sensor7'])
print('Denoising signals...')
train_df, test_df = denoise(train_df, test_df, filtered_signals)
print('Signals denoised')

plots.plot_engine_signals(1, train_df, filtered_signals)
plots.plot_superposed_engine_signals(range(1, 101), train_df, filtered_signals)

plots.plot_correlation_map(train_df, filtered_signals + ['time_to_failure'])
plots.plot_pca_variance_contribution(train_df, filtered_signals)

# Perform PCA
n_pc_components = 2
pc_columns = ['PC{}'.format(i+1) for i in range(n_pc_components)]
scaler = StandardScaler()
scaler.fit(train_df[filtered_signals])
pca = PCA(n_components=2)
train_pca = pd.DataFrame(pca.fit_transform(scaler.transform(train_df[filtered_signals])), columns=pc_columns)
test_pca = pd.DataFrame(pca.transform(scaler.transform(test_df[filtered_signals])), columns=pc_columns)
train_pca['time_to_failure'] = train_df.time_to_failure.reset_index(drop=True)
test_pca['time_to_failure'] = test_df.time_to_failure.reset_index(drop=True)
plots.plot_3d_lifetime_paths(train_pca)


feature_engineering_functions = [
    (feature_engineering.delta_signal, {'n_lag': 20}),
    (feature_engineering.rolling_mean, {'time_window_length': 20}),
    (feature_engineering.mean_derivative, {'time_window_length': 20}),
    (feature_engineering.rolling_max, {'time_window_length': 20}),
    (feature_engineering.rolling_min, {'time_window_length': 20}),
    (feature_engineering.rolling_abs_energy, {'time_window_length': 20}),
    (feature_engineering.rolling_abs_sum_of_changes, {'time_window_length': 20}),
    (feature_engineering.rolling_std, {'time_window_length': 20}),
    (feature_engineering.time_reversal_asymmetry, {'time_window_length': 20, 'n_lag': 5}),
    (feature_engineering.log, {}),
    (feature_engineering.diff_signal, {'n_lag': 20}),
    (feature_engineering.gamma_signal, {'n_lag': 20})
]

# TODO: fix functions
train_df, test_df = feature_engineering.feature_engineer(train_df,
                                                         test_df,
                                                         filtered_signals,
                                                         feature_engineering_functions,
                                                         silent=False,
                                                         drop_na=True)
print('Feature engineering complete')
train_df.to_csv('feature_engineered_train.csv')
test_df.to_csv('feature_engineered_test.csv')

train_df = pd.read_csv('feature_engineered_train.csv').set_index('id')
test_df = pd.read_csv('feature_engineered_test.csv').set_index('id')

X, y = train_df.drop('time_to_failure', axis=1), train_df.time_to_failure
X_test, y_test = test_df.drop('time_to_failure', axis=1), test_df.time_to_failure


# Classification fitting and evaluation
classification_estimator = classification_best_model['estimator']
classification_features = classification_best_model['feature_subset']

classification_estimator.fit(X[classification_features], y.apply(lambda x: 1 if x <= 30 else 0))
classification_predictions = classification_estimator.predict(X_test[classification_features])
classification_predictions_proba = classification_estimator.predict_proba(X_test[classification_features])

plots.plot_classification_report(y_test.apply(lambda x: 1 if x <= 30 else 0), classification_predictions, y_test)

# Regression fitting and evaluation
