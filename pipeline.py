import pandas as pd
import numpy as np
from random import shuffle

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest

import plots
from preprocessing import drop_constant_signals, generate_labels, denoise
from feature_engineering import feature_engineer, lag, time_reversal_asymmetry, rolling_mean


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
print('Filtered signals :', filtered_signals)

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

feature_engineering_functions = {
    lag: {'n_lag': 10},
    rolling_mean: {'time_window_length': 20}
    # time_reversal_asymmetry: {'time_window_length': 20, 'n_lag': 5},
}

train_df, test_df = feature_engineer(train_df, test_df, filtered_signals, feature_engineering_functions, silent=False)
print('Feature engineering complete')


def split_engines_for_CV(train, n_folds=5):
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


splits = split_engines_for_CV(train_df, 5)


pipe = Pipeline(steps=[('kbest', SelectKBest()),
                       ('scaler', MinMaxScaler()),
                       ('pca', PCA()),
                       ('randomforest', RandomForestRegressor())])

param_grid = {
    'kbest__k': list(map(lambda x: min(x, train_df.shape[1]), [20, 30, 40, 50])),
    'pca__n_components': [5, 10, 20],
    'randomforest__n_estimators': [10, 20, 40, 100],
    'randomforest__max_depth': [3, 8, None]
}

X, y = train_df.drop('time_to_failure', axis=1), train_df.time_to_failure

grid_search = RandomizedSearchCV(pipe,
                                 param_grid,
                                 scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'],
                                 refit='neg_mean_absolute_error')
grid_search.fit(X, y)
print(grid_search.best_params_)
cv_results = pd.DataFrame(grid_search.cv_results_)


X_test, y_test = test_df.drop('time_to_failure', axis=1), test_df.time_to_failure
pipe = pipe.set_params(**grid_search.best_params_)
pipe.fit(X, y)
predictions = pipe.predict(X_test)

trues_predictions = pd.DataFrame(zip(test_df.time_to_failure, predictions), index=test_df.index)
test_df['prediction'] = predictions
