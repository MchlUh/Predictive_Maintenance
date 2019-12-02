import pandas as pd
import numpy as np
from sklearn.feature_selection import variance_threshold
from statsmodels.tsa.filters import hp_filter


# Drop constant columns providing no information
# The fitting is done on train dataset. The exact same operation is performed on the test data
# Notice that no "training" will ever be performed on test dataset
# Only the same "transform" operations that were learned on training dataset will be performed on test data
# This should be respected in production.
def drop_constant_signals(train, test, cols, threshold=0.01):
    var = variance_threshold.VarianceThreshold(threshold)
    train = pd.DataFrame(var.fit_transform(train), index=train.index)
    test = pd.DataFrame(var.transform(test), index=test.index)

    columns_filter = var.get_support().tolist()
    filtered_columns = list(filter(None, [cols[i] if columns_filter[i] else None for i in range(len(cols))]))
    train.columns, test.columns = filtered_columns, filtered_columns

    filtered_sensor_columns = list(filter(None, [column if column.startswith("sensor")
                                                 else None for column in train.columns]))
    return train, test, filtered_sensor_columns


def generate_labels(train, test, test_label_file_name):
    train["time_to_failure"] = pd.Series()
    for engine_id in train.id.unique():
        train.loc[train.id == engine_id, "time_to_failure"] = train[train.id == engine_id]["time"].max() \
                                                                    - train[train.id == engine_id]["time"]
    train.time_to_failure = train.time_to_failure.astype(np.int32)
    train = train.drop(columns="time")

    label_file = open(test_label_file_name, "r")
    test_labels = [int(line) for line in label_file]
    label_file.close()
    test["time_to_failure"] = pd.Series()
    for engine_id in test.id.unique():
        test.loc[test.id == engine_id, "time_to_failure"] = test_labels[int(engine_id) - 1] \
                                                                  + test.loc[test.id == engine_id]["time"].max() \
                                                                  - test.loc[test.id == engine_id]["time"]
    test = test.drop(columns="time")
    test.time_to_failure = test.time_to_failure.astype(np.int32)

    return train, test


def denoise(train, test, sensor_columns, lamb=1600):
    # De-noise signals with Hodrick-Prescott filter
    for sensor in sensor_columns:
        for engine_id in train.id.unique():
            _, train.loc[train.id == engine_id, sensor] = \
                hp_filter.hpfilter(train.loc[train.id == engine_id, sensor], lamb)
        for engine_id in test.id.unique():
            _, test.loc[test.id == engine_id, sensor] = \
                hp_filter.hpfilter(test.loc[test.id == engine_id, sensor], lamb)
    return train, test
