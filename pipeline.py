import pandas as pd
import numpy as np
from random import shuffle

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

import plots
import feature_engineering
from preprocessing import drop_constant_signals, generate_labels, denoise
from metrics import mean_absolute_percentage_error, mean_custom_error

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

print('Dropping constant signals...')
train_df, test_df, filtered_signals = drop_constant_signals(train_df, test_df, test_df.columns)
print('Filtered signals :', *filtered_signals)

label_file = "CMAPSSData/CMAPSSData/RUL_FD00{}.txt".format(dataset_number)
train_df, test_df = generate_labels(train_df, test_df, label_file)
print('Generated labels')

print('Denoising signals...')
train_df, test_df = denoise(train_df, test_df, filtered_signals)
print('Signals denoised')

plots.plot_engine_signals(1, train_df, filtered_signals)
plots.plot_superposed_engine_signals(range(1, 101), train_df, filtered_signals[:1])

plots.plot_correlation_map(train_df, filtered_signals + ['time_to_failure'])
plots.plot_pca_variance_contribution(train_df, filtered_signals)
n_pc_components = 2
pc_columns = ['PC{}'.format(i+1) for i in range(n_pc_components)]

# Perform PCA
scaler = MinMaxScaler()
scaler.fit(train_df[filtered_signals])
train_df[filtered_signals] = scaler.transform(train_df[filtered_signals])
test_df[filtered_signals] = scaler.transform(test_df[filtered_signals])

pca = PCA(n_components=2)
train_pca = pd.DataFrame(pca.fit_transform(train_df[filtered_signals]), columns=pc_columns, index=train_df.index)
test_pca = pd.DataFrame(pca.transform(test_df[filtered_signals]), columns=pc_columns, index=test_df.index)
for pc_col in pc_columns:
    train_df[pc_col], test_df[pc_col] = train_pca[pc_col], test_pca[pc_col]
plots.plot_3d_lifetime_paths(train_df)

train_df, test_df = train_df.drop(filtered_signals, axis=1), test_df.drop(filtered_signals, axis=1)


feature_engineering_functions = [
    (feature_engineering.lag, {'n_lag': 10}),
    (feature_engineering.rolling_mean, {'time_window_length': 20}),
    (feature_engineering.mean_derivative, {'time_window_length': 20}),
    (feature_engineering.rolling_max, {'time_window_length': 20}),
    (feature_engineering.rolling_min, {'time_window_length': 20}),
    (feature_engineering.rolling_abs_energy, {'time_window_length': 20}),
    (feature_engineering.rolling_abs_sum_of_changes, {'time_window_length': 20}),
    (feature_engineering.rolling_variance, {'time_window_length': 20}),
    (feature_engineering.time_reversal_asymmetry, {'time_window_length': 20, 'n_lag': 5})
]

train_df, test_df = feature_engineering.feature_engineer(train_df,
                                                         test_df,
                                                         pc_columns,
                                                         feature_engineering_functions,
                                                         silent=False)
print('Feature engineering complete')


def split_engines_for_cv(train, n_folds=5):
    """
    Splits engines into n_folds folds for cross-validation
    :param train: training set
    :param n_folds: number of folds
    :return: list of lists containing engine ids in every fold
    """
    engine_ids = list(set(train.index))
    shuffle(engine_ids)
    engine_splits = []
    split_length = len(engine_ids)//n_folds
    for i in range(n_folds - 1):
        engine_splits.append(engine_ids[i*split_length:(i+1)*split_length])
    engine_splits.append(engine_ids[(n_folds-1)*split_length:])
    return engine_splits


X, y = train_df.drop('time_to_failure', axis=1), train_df.time_to_failure

pipe = Pipeline(steps=[#('kbest', SelectKBest()),
                       ('scaler', MinMaxScaler()),
                       ('pca', PCA()),
                       ('xgb', xgb.XGBRegressor())])

param_grid = {
   # 'kbest__k': list(map(lambda x: min(x, train_df.shape[1] - 1), [20, 30, 40, 50])),
    'pca__n_components': [5, 10, min(20, X.shape[1])],
    'xgb__n_estimators': [10, 20, 40, 100],
    'xgb__max_depth': [2, 3, 4]
}

splits = split_engines_for_cv(train_df, 5)  # TODO: feed splits to GridSearchCV
grid_search = RandomizedSearchCV(pipe,
                                 param_grid,
                                 scoring='neg_mean_absolute_error',
                                 n_iter=3
                                 )
grid_search.fit(X, y)
print(grid_search.best_params_)
cv_results = pd.DataFrame(grid_search.cv_results_)

# Final predictions on test set
X_test, y_test = test_df.drop('time_to_failure', axis=1), test_df.time_to_failure
pipe = pipe.set_params(**grid_search.best_params_)
pipe.fit(X, y)
predictions = pipe.predict(X_test)

trues_vs_predictions = pd.DataFrame(zip(y_test, predictions), index=test_df.index, columns=['True', 'Prediction'])

print('MAE {mae} RMSE {rmse} Mean Custom Error {custom_error}'.format(
    rmse=mean_squared_error(y_test, predictions),
    mae=mean_absolute_error(y_test, predictions),
    custom_error=mean_custom_error(y_test, predictions)))

plots.plot_error_repartition(y_test, predictions)
