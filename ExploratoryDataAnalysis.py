import pandas as pd
import numpy as np
import random
from datetime import datetime

from sklearn.feature_selection import variance_threshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.filters import hp_filter
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from statsmodels.graphics.correlation import plot_corr
from collections import namedtuple
from copy import deepcopy

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

dataset_number = 1

# Read Data
columns = ["id", "time", "op1", "op2", "op3"]
sensor_columns = ["sensor{}".format(i+1) for i in range(21)]
columns.extend(sensor_columns)
train_df = pd.read_csv("CMAPSSData/CMAPSSData/train_FD00{}.txt".format(dataset_number), delimiter=" ", index_col=None, header=None)
test_df = pd.read_csv("CMAPSSData/CMAPSSData/test_FD00{}.txt".format(dataset_number), delimiter=" ", index_col=None, header=None)
train_df, test_df = train_df.dropna(axis=1), test_df.dropna(axis=1)
train_df.columns, test_df.columns = columns, columns


# Drop constant columns providing no information
# The fitting is done on train dataset. The exact same operation is performed on the test data
# Notice that no "training" will ever be performed on test dataset
# Only the same "transform" operations that were learned on training dataset will be performed on test data
# This should be respected in production.
def apply_var_threshold(train, test, threshold=0.01, cols=tuple(train_df.columns)):
    var = variance_threshold.VarianceThreshold(threshold)
    train, test = pd.DataFrame(var.fit_transform(train)), pd.DataFrame(var.transform(test))

    columns_filter = var.get_support().tolist()
    filtered_columns = list(filter(None, [cols[i] if columns_filter[i] else None for i in range(len(cols))]))
    train.columns, test.columns = filtered_columns, filtered_columns

    sensor_cols = list(filter(None, [column if column.startswith("sensor") else None for column in train.columns]))
    return train, test, sensor_cols


train_df, test_df, sensor_columns = apply_var_threshold(train_df, test_df)

train_df.id, train_df.time = train_df.id.astype(int), train_df.time.astype(int)
test_df.id, test_df.time = test_df.id.astype(int), test_df.time.astype(int)

sensors = tuple(sensor for sensor in train_df.columns if "sensor" in sensor)


def plot_engine_signals(engine_id, df=train_df, signals=sensors):
    [plt.plot(df.loc[df.id == engine_id, "time_to_failure"],
              df.loc[df.id == engine_id, signal]) for signal in signals]
    plt.show()


def plot_superposed_engine_signals(engine_ids, df=train_df, signals=("PC1", "PC2")):
    [[plt.plot(df.loc[df.id == id, "time_to_failure"],
              df.loc[df.id == id, signal]) for signal in signals] for id in engine_ids]
    plt.xlim(0, 200)
    plt.show()


# Generate labels
train_df["time_to_failure"] = pd.Series()
for engine_id in train_df.id.unique():
    train_df.loc[train_df.id == engine_id, "time_to_failure"] = train_df[train_df.id == engine_id]["time"].max()\
                                                                - train_df[train_df.id == engine_id]["time"]
train_df.time_to_failure = train_df.time_to_failure.astype(int)
train_df = train_df.drop(columns="time")

label_file = open("CMAPSSData/CMAPSSData/RUL_FD00{}.txt".format(dataset_number), "r")
test_labels = [int(line) for line in label_file]
label_file.close()
for engine_id in test_df.id.unique():
    test_df.loc[test_df.id == engine_id, "time_to_failure"] = test_labels[engine_id - 1] \
                                                              + test_df.loc[test_df.id == engine_id, "time"].max() \
                                                              - test_df.loc[test_df.id == engine_id, "time"]
test_df = test_df.drop(columns="time")
test_df.time_to_failure = test_df.time_to_failure.astype(int)


# De-noise signals
for sensor in sensor_columns:
    for engine_id in train_df.id.unique():
        _, train_df.loc[train_df.id == engine_id, sensor] =\
            hp_filter.hpfilter(train_df.loc[train_df.id == engine_id, sensor], 1600)
    for engine_id in test_df.id.unique():
        _, test_df.loc[test_df.id == engine_id, sensor] =\
            hp_filter.hpfilter(test_df.loc[test_df.id == engine_id, sensor], 1600)


# normalize signals
scaler = MinMaxScaler()
train_df[sensor_columns] = scaler.fit_transform(train_df[sensor_columns])
test_df[sensor_columns] = scaler.transform(test_df[sensor_columns])


# Plot correlation map -- sensor signals are highly correlated !
cmap = sns.diverging_palette(5, 250, as_cmap=True)
corr = train_df[sensor_columns].corr()
# sns.heatmap(corr,
#            xticklabels=corr.columns,
#            yticklabels=corr.columns,
#            cmap=cmap)
plot_corr(corr, xnames=sensor_columns, ynames=sensor_columns, title='Sensor Correlations')
plt.show()


# Let's perform dimensionality reduction
# Perform PCA with max number of components and choose the right number of components
def plot_pca_variance_contribution(df, cols):
    pca_plot = PCA(n_components=len(cols))
    pca_plot.fit(df[cols])
    y = [0]
    y.extend(np.cumsum(pca_plot.explained_variance_ratio_))
    plt.plot(y)
    plt.title("Cumulative variance contribution")
    plt.xlabel("Number of principal components")
    plt.xticks(list(range(1, 10)))
    plt.ylim(y[1], 1)
    plt.xlim(1, len(cols))
    plt.grid()
    plt.show()


plot_pca_variance_contribution(train_df, sensor_columns)

n_principal_components = 2
pca = PCA(n_components=n_principal_components)
principal_components = ["PC{}".format(i + 1) for i in range(n_principal_components)]
for i in range(n_principal_components):
    train_df[principal_components[i]] = pd.Series()
    test_df[principal_components[i]] = pd.Series()

train_df[principal_components] = pca.fit_transform(train_df[sensor_columns])
test_df[principal_components] = pca.fit_transform(test_df[sensor_columns])


def time_reversal_asymmetry(x, lag):
    n = len(x)
    x = np.asarray(x)
    if 2 * lag >= n:
        return 0
    else:
        one_lag = np.roll(x, -lag)
        two_lag = np.roll(x, 2 * -lag)
        return np.mean((two_lag * two_lag * one_lag - one_lag * x * x)[0:(n - 2 * lag)])


# Let's perform some feature engineering to add information for the model
# We'll begin simply, by adding mean and variance over the 20 last signal observations for every engine
for sensor in sensors:
    train_df[sensor + "_mean"], test_df[sensor + "_mean"] = pd.Series(), pd.Series()  # TODO : Rename Time_reversal
    train_df[sensor + "_tra"], test_df[sensor+"_tra"] = pd.Series(), pd.Series()

t_begin_feature_engineering = datetime.now()
window_length = 30
engineered_columns = []
for sensor in sensors:
    for engine_id in train_df.id.unique():
        col_name = sensor + "_mean"
        train_df.loc[train_df.id == engine_id, col_name] =\
            train_df.loc[train_df.id == engine_id, sensor].rolling(window_length).mean()
        test_df.loc[test_df.id == engine_id, col_name] =\
            test_df.loc[test_df.id == engine_id, sensor].rolling(window_length).mean()
        engineered_columns.append(col_name)

        col_name = sensor + "_tra"
        train_df.loc[train_df.id == engine_id, col_name] =\
            train_df.loc[train_df.id == engine_id, sensor].rolling(window_length)\
                .apply(lambda x: time_reversal_asymmetry(x, window_length//4), raw=False)
        test_df.loc[test_df.id == engine_id, col_name] =\
            test_df.loc[test_df.id == engine_id, sensor].rolling(window_length)\
               .apply(lambda x: time_reversal_asymmetry(x, window_length//4), raw=False)
        engineered_columns.append(col_name)

t_end_feature_engineering = datetime.now()
print("Feature engineering took {time} for {num} columns, hence {avg} per column".format(
      time=t_end_feature_engineering-t_begin_feature_engineering,
      num=len(sensors),
      avg=(t_end_feature_engineering-t_begin_feature_engineering)/len(sensors)))


# Rolling window introduced NaN values to the dataset
# Indeed the first 20 points of each signal cannot have aggregate function computed
# We would like to avoid NaN values in the dataframe. Moreover, these values correspond
# to beginning of engine lifetimes,
# Let's get rid of the lines containing them.
train_df, test_df = train_df.dropna(axis=0), test_df.dropna(axis=0)


# let's normalize those new features
principal_components = list(train_df.columns)
principal_components.remove("id")
principal_components.remove("time_to_failure")
scaler = MinMaxScaler()
train_df[principal_components] = scaler.fit_transform(train_df[principal_components])
test_df[principal_components] = scaler.transform(test_df[principal_components])


train_df, test_df = train_df.drop(columns=engineered_columns), test_df.drop(columns=engineered_columns)


# normalize PC signals
scaler = MinMaxScaler()
train_df[principal_components] = scaler.fit_transform(train_df[principal_components])
test_df[principal_components] = scaler.transform(test_df[principal_components])


# Plotting time !
# Let's see how signals evolve during the lifetime of these engines
def plot_3d_lifetime_paths():
    fig = plt.figure().gca(projection='3d')
    fig.scatter(train_df.PC1,
                train_df.PC2,
                train_df.time_to_failure,
                s=0.02,
                c=train_df.time_to_failure,
                cmap='magma')
    fig.set_xlabel('PC1')
    fig.set_ylabel('PC2')
    fig.set_zlabel('time_to_failure')
    fig.set_zlim3d(0, 250)
    fig.set_title("Engines lifetime paths")
    plt.show()


def plotly_3d_lifetime_paths():
    fig = go.Figure(data=[go.Scatter3d(
        x=train_df[train_df.time_to_failure < 150].PC1,
        y=train_df[train_df.time_to_failure < 150].PC2,
        z=train_df[train_df.time_to_failure < 150].time_to_failure,
        mode='markers',
        marker=dict(
            size=1.5,
            color=train_df.time_to_failure,  # set color to an array/list of desired values
            colorscale='Magma',  # choose a colorscale
            opacity=0.5
        )
    )])
    fig.update_layout(scene=dict(
        zaxis=dict(range=[0, 100], ), ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))
    fig.write_html("lala.html", auto_open=True)


# PC1 signals seem to gather at 1 for end of lifetime
# and PC2 signals gather somewhere around 0.5 for beginning of life.
# Furthermore, the data points are more spaced out at end of life (derivatives are higher!)
# We might want to add this information to the dataset by computing derivatives or variance
# All of this will give the model insight about health conditions.
plot_3d_lifetime_paths()
plotly_3d_lifetime_paths

# Save PCA dataframes
train_df.to_csv("train_2PCs.csv")
test_df.to_csv("test_2PCs.csv")


# We can see that 4 components instead of 6 are sufficient to represent 99.99% of the variance
# n_features = 3
# final_features = ["Feat{}".format(i + 1) for i in range(n_features)]
# pca = PCA(n_features)
# for i in range(n_features):
#     train_df[final_features[i]] = pd.Series()
#     test_df[final_features[i]] = pd.Series()
#
# train_df[final_features] = pca.fit_transform(train_df[pc_columns])
# test_df[final_features] = pca.transform(test_df[pc_columns])
# train_df, test_df = train_df.drop(columns=pc_columns), test_df.drop(columns=pc_columns)
#
#
# # Let's rescale the output features
# scaler = MinMaxScaler()
# train_df[final_features] = scaler.fit_transform(train_df[final_features])
# test_df[final_features] = scaler.transform(test_df[final_features])
#
# # Save preprocessed dataframes
# train_df.to_csv("train_processed_4features.csv")
# test_df.to_csv("test_processed_4features.csv")


# We have our final features. We performed feature selection and dimensionality reduction
# on an initial pool of 21 sensors. We added dimension on each data point by incorporating past behavior
# We have now 4 final features with which to build a model. Those features are normalized and orthogonal.
# It's time to build our first model.


# We will evaluate the model using cross-validation
# We need a custom CV function to perform CV,
# because the trained models cannot see a part of other engine's time series

# Let's add past values of signals


# train_df, test_df = pd.read_csv("train_processed.csv"), pd.read_csv("test_processed.csv")
final_features = list(train_df.columns)
final_features.remove("id")
final_features.remove("time_to_failure")


def cross_validation(kfold, model_parameters):
    if len(train_df.id.unique()) % kfold:
        # Here we assume that number of engines is not prime #todo: clean dis shit
        # delete this condition if number of train units is not trivial
        print("Please choose k as a divisor of {} for kfold cross validation".format(len(train_df.id.unique())))
        return

    def train_test_engines_split(k, engine_ids):
        ids = deepcopy(engine_ids)
        random.shuffle(ids)
        return [ids[len(engine_ids) // k * j:len(engine_ids) // k * (j + 1)] for j in range(k)]

    splits = train_test_engines_split(kfold, train_df.id.unique())

    mae = [0] * kfold
    rmse = [0] * kfold

    for i in range(kfold):
        # TRAIN MODEL
        gbm = xgb.XGBRegressor(max_depth=model_parameters["max_depth"],
                               learning_rate=model_parameters["eta"],
                               n_estimators=model_parameters["n_estimators"],
                               colsample_bytree=model_parameters["colsample_bytree"],
                               gamma=model_parameters["gamma"],
                               min_child_weight=model_parameters["min_child_weight"],
                               subsample=model_parameters["subsample"],
                               # eval_metric=model_parameters["eval_metric"],
                               alpha=model_parameters["alpha"],
                               objective="reg:squarederror")

        gbm.fit(train_df.loc[~train_df.id.isin(splits[i]), final_features],  # Features
                train_df.loc[~train_df.id.isin(splits[i])].time_to_failure)  # Label

        predictions = gbm.predict(train_df.loc[train_df.id.isin(splits[i]), final_features])
        rmse[i] = np.sqrt(mean_squared_error(predictions,
                                             train_df.loc[train_df.id.isin(splits[i]), "time_to_failure"]))

        mae[i] = mean_absolute_error(predictions,
                                     train_df.loc[train_df.id.isin(splits[i]), "time_to_failure"])
        xgb.plot_importance(gbm)
        plt.show()
    return mae, rmse


params = {'colsample_bytree': 1,
          'eta': 0.3,
          'gamma': 6,
          'max_depth': 4,
          'min_child_weight': 4,
          'n_estimators': 500,
          'subsample': 1,
          "alpha": 0,
          "eval_metric": "rmse"}

mae, rmse = cross_validation(10, params)
print("CV Results \nMAE mean : {0}, MAE variance : {1}\nRMSE mean : {2}, RMSE variance : {3}"
      .format(np.mean(mae), np.var(mae), np.mean(rmse), np.var(rmse)))

# Time for the final test on our model : its performance on the test set.

gbm = xgb.XGBRegressor(max_depth=params["max_depth"],
                       learning_rate=params["eta"],
                       n_estimators=params["n_estimators"],
                       colsample_bytree=params["colsample_bytree"],
                       gamma=params["gamma"],
                       min_child_weight=params["min_child_weight"],
                       subsample=params["subsample"],
                       # eval_metric=params["eval_metric"],
                       alpha=params["alpha"],
                       objective="reg:squarederror")

gbm.fit(train_df[final_features],  # Features
        train_df.time_to_failure)  # Label

predictions = gbm.predict(test_df[final_features])
test_df["prediction"] = predictions
final_rmse = np.sqrt(mean_squared_error(predictions,
                                     test_df["time_to_failure"]))

final_mae = mean_absolute_error(predictions,
                             test_df["time_to_failure"])
xgb.plot_importance(gbm)
plt.show()
print(str(final_mae), str(final_rmse))


def plot_prediction(engine_id):
    plt.plot(test_df.loc[test_df.id == engine_id, "time_to_failure"],
             test_df.loc[test_df.id == engine_id, "prediction"])
    plt.plot(test_df.loc[test_df.id == engine_id, "time_to_failure"],
             test_df.loc[test_df.id == engine_id, "time_to_failure"])
    plt.show()


