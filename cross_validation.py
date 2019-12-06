from random import shuffle
import itertools
import pandas as pd
import xgboost
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
import numpy as np
import warnings
import copy

warnings.simplefilter(action='ignore', category=FutureWarning)


def split_engines_for_cv(train, n_folds=5, fifty_percent_splits=False):
    """
    Splits engines into n_folds folds for cross-validation
    Returns list of frame indices for each fold
    :param train: training set, indexed by engine id
    :param n_folds: number of folds
    :return: list of lists containing engine ids in every fold
    """
    engine_ids = list(set(train.index))
    train = train.copy().reset_index()

    if not fifty_percent_splits:
        shuffle(engine_ids)
        engine_splits = []
        split_length = len(engine_ids)//n_folds
        for i in range(n_folds - 1):
            engine_splits.append(engine_ids[i*split_length:(i+1)*split_length])
        engine_splits.append(engine_ids[(n_folds-1)*split_length:])

        split_rows = [train.id.isin(engine_splits[i]) for i in range(n_folds)]

        split_indices = [[indice for indice, is_in_split_i in enumerate(split_rows[i]) if is_in_split_i]
                         for i in range(n_folds)]

        cv_folds = [  # Train instances for fold i
                    (np.array(list(itertools.chain.from_iterable(split_indices[:i] + split_indices[i+1:]))),
                      # Test instances for fold i
                     (np.array(split_indices[i]))) for i in range(n_folds)]

        return cv_folds

    else:  # split dataset multiple times by half and return folds
        cv_folds = []
        for i in range(n_folds//2 + 1):
            shuffle(engine_ids)
            split_length = len(engine_ids) // 2
            engine_splits = [copy.deepcopy(engine_ids[:split_length]), copy.deepcopy(engine_ids[split_length:])]
            split_rows = [train.id.isin(engine_splits[i]) for i in range(2)]
            split_indices = [[indice for indice, is_in_split_i in enumerate(split_rows[i]) if is_in_split_i]
                             for i in range(2)]
            cv_folds.append([np.array(copy.deepcopy(split_indices[0])), np.array(copy.deepcopy(split_indices[1]))])
            cv_folds.append([np.array(copy.deepcopy(split_indices[1])), np.array(copy.deepcopy(split_indices[0]))])

        return cv_folds[:n_folds]


if __name__ == '__main__':
    train_df = pd.read_csv('feature_engineered_train.csv').set_index('id')
    test_df = pd.read_csv('feature_engineered_test.csv').set_index('id')

    X_train, y_train = train_df.drop('time_to_failure', axis=1), train_df.time_to_failure
    X_test, y_test = test_df.drop('time_to_failure', axis=1), test_df.time_to_failure

    xgb_model = xgboost.XGBRegressor(max_depth=3, colsample_bytree=0.8, subsample=0.8, n_estimators=100)

    custom_cross_val = cross_validate(xgb_model, X_train, y_train, cv=split_engines_for_cv(train_df, 2, True),
                                      scoring='neg_mean_absolute_error')
    print('Custom CV mean test score:', -np.mean(custom_cross_val['test_score']))

    normal_cross_val = cross_validate(xgb_model, X_train, y_train, cv=2, scoring='neg_mean_absolute_error')
    print('Normal CV mean test score:', -np.mean(normal_cross_val['test_score']))

    xgb_model.fit(X_train, np.log(y_train+1))
    mean_absolute_error_test = mean_absolute_error(np.exp(xgb_model.predict(X_test)) - 1, y_test)
    print(mean_absolute_error_test)
