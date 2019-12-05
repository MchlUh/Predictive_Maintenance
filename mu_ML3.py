import pandas as pd
import numpy as np

from Assignment3.Predictive_Maintenance.preprocessing import drop_constant_signals, generate_labels, denoise
from Assignment3.Predictive_Maintenance.plots import *
from sklearn.linear_model import LinearRegression

from Assignment3.Predictive_Maintenance.feature_engineering import feature_engineer, lag, time_reversal_asymmetry, rolling_mean


# CMAPSS dataset number
dataset_number = 1

# Read Data
columns = ["id", "time", "op1", "op2", "op3"]
sensor_columns = ["sensor{}".format(i+1) for i in range(21)]
columns.extend(sensor_columns)
train_df = pd.read_csv("Assignment3/Predictive_Maintenance/CMAPSSData/CMAPSSData/train_FD00{}.txt".format(dataset_number),
                       delimiter=" ", index_col=None, header=None)
test_df = pd.read_csv("Assignment3/Predictive_Maintenance/CMAPSSData/CMAPSSData/test_FD00{}.txt".format(dataset_number),
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

label_file = "Assignment3/Predictive_Maintenance/CMAPSSData/CMAPSSData/RUL_FD00{}.txt".format(dataset_number)
train_df, test_df = generate_labels(train_df, test_df, label_file)
print('Generated labels')

print('Denoising signals...')
train_df, test_df = denoise(train_df, test_df, filtered_signals)
print('Signals denoised')

plot_engine_signals(1, train_df, filtered_signals)
plot_superposed_engine_signals(range(1, 101), train_df, filtered_signals[:1])

plot_correlation_map(train_df, filtered_signals + ['time_to_failure'])
plot_pca_variance_contribution(train_df, filtered_signals)


engines = train_df['id'].unique()
intercepts = np.zeros(len(engines))
coeffs = np.zeros(len(engines))

train_df.loc[1]


for engine in engines:
    df_sub = train_df.copy(deep=True)
    df_sub = df_sub.loc[engine]
    df_sub[list(filtered_signals)] = df_sub[list(filtered_signals)].apply(lambda x: np.log(x+0.1))
    df_sub.reset_index(inplace=True, drop=True)
    for signal in filtered_signals:
        lm = LinearRegression().fit(df_sub[signal].values.reshape(-1, 1), df_sub.index.values)
    intercepts[int(engine)-1] = lm.intercept_
    coeffs[int(engine)-1] = lm.coef_[0]
