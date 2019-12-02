import pandas as pd
import numpy as np

from preprocessing import drop_constant_signals, generate_labels, denoise
import plots

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
train_df, test_df = train_df.set_index('id', drop=False), test_df.set_index('id', drop=False)

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
    rolling_mean: {'time_window_length': 20},
    time_reversal_asymmetry: {'time_window_length': 20, 'n_lag': 5}
}

train_df, test_df = feature_engineer(train_df, test_df, filtered_signals, feature_engineering_functions)
