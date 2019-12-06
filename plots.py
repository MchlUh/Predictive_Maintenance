import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import pandas as pd
from collections import OrderedDict

from statsmodels.graphics.correlation import plot_corr
from statsmodels.tsa.filters import hp_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from seaborn import heatmap

plt.style.use('seaborn')


def plot_denoising_process(signal):
    fig = plt.figure()
    signal = signal.reset_index(drop=True)
    plt.ylim((signal.min(), signal.max()))
    plt.plot(signal, label='Noisy signal')
    plt.plot(signal.rolling(30).mean(), label='Rolling mean filter', c='green')
    plt.plot(signal.ewm(alpha=0.05, adjust=True).mean(), label='Exponential smoothing', c='orange')
    plt.plot(pd.Series(hp_filter.hpfilter(signal, 800)[1]), label='Hodrick-Prescott filter', c='red')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel(signal.name)
    plt.title('Comparison of filtering methods')
    plt.show()


def plot_engine_signals(engine_id, df, signals, scale_signals=True):
    df = df.copy()
    if scale_signals:
        scaler = StandardScaler()
        df.loc[:, signals] = scaler.fit_transform(df.loc[:, signals])

    if 'time_to_failure' in df.columns:
        [plt.plot(df.loc[engine_id, "time_to_failure"],
                  df.loc[engine_id, signal]) for signal in signals]
        plt.xlabel('Time to Failure')
        plt.xlim(0, df.time_to_failure.max())
        ax = plt.gca()
        ax.set_xlim(ax.get_xlim()[::-1])  # Reverse x axis (time to failure ends at 0)

    else:
        [plt.plot(df.loc[engine_id, "time"],
                  df.loc[engine_id, signal]) for signal in signals]
        plt.xlabel('Time')

    plt.title("Evolution of {n_signals} signals for engine {engine_id}".format(
        n_signals=len(signals), engine_id=engine_id)
    )
    plt.show()


def plot_superposed_engine_signals(engine_ids, df, signals, show=True):
    fig = plt.figure(figsize=(15,10))
    n_plot_columns = len(signals) if len(signals)<3 else 3
    n_plot_rows = 1 if len(signals) <= 3 else (len(signals)//3 if len(signals) % 3 == 0 else len(signals)//3 + 1)
    plot_number = 1

    for signal in signals:
        fig.add_subplot(n_plot_rows, n_plot_columns, plot_number)
        [plt.plot(df.loc[engine_id, "time_to_failure"], df.loc[engine_id, signal], linewidth=.4)
         for engine_id in engine_ids]
        plt.xlabel('Time to Failure')
        plt.ylabel(signal)
        plt.xlim(0, df.time_to_failure.max())
        ax = plt.gca()
        ax.set_xlim(ax.get_xlim()[::-1])
        plot_number += 1
    fig.subplots_adjust(top=0.8)
    if show:
        plt.show()


def plot_pca_variance_contribution(df, cols):
    pca_plot = PCA(n_components=len(cols))
    pca_plot.fit(df[cols])
    y = [0]
    y.extend(np.cumsum(pca_plot.explained_variance_ratio_))
    plt.plot(y)
    plt.title("Cumulative variance contribution")
    plt.xlabel("Number of principal components")
    plt.xticks(list(range(1, len(cols))))
    plt.ylim(y[1], 1)
    plt.xlim(1, len(cols))
    plt.show()


def plot_correlation_map(train, sensor_columns):
    corr = train[sensor_columns].corr()
    plot_corr(corr, xnames=sensor_columns, ynames=sensor_columns, title='Sensor Correlations')
    plt.show()


def plot_3d_lifetime_paths(train):
    if 'PC1' not in train.columns:
        print('Please perform PCA with 2 principal components before calling plot_3d_lifetime_paths')
        return
    fig = plt.figure().gca(projection='3d')
    fig.scatter(train.PC1,
                train.PC2,
                train.time_to_failure,
                s=0.5,
                c=train.time_to_failure,
                cmap='magma')
    fig.set_xlabel('PC1')
    fig.set_ylabel('PC2')
    fig.set_zlabel('time_to_failure')
    fig.set_zlim3d(0, 150)
    fig.set_title("Engines lifetime paths with 2 principal components")
    plt.show()


def plot_error_repartition(trues, preds, show=True):
    plt.grid()
    errors = OrderedDict.fromkeys(sorted(set(trues)))
    for i in set(trues):
        errors[i] = []
    [errors[true].append(np.abs(pred-true)) for true, pred in zip(trues, preds)]
    plt.plot(list(errors.keys()), list(map(np.mean, errors.values())), label='Mean error')
    plt.plot(list(errors.keys()), list(map(np.max, errors.values())), label='Max error')
    plt.plot(list(errors.keys()), list(map(np.median, errors.values())), label='Median error')
    plt.xlabel('True TTF values')
    plt.ylabel('Absolute error')
    plt.legend()
    plt.title("Absolute error repartition")
    if show:
        plt.show()


def plot_residuals(trues, preds, show=True):
    plt.grid()
    plt.scatter(trues, preds-trues, s=3)
    plt.plot(range(151), list(map(lambda x: x/5, range(151))), label='20% overestimation', c='darkred', linestyle='--')
    plt.plot(range(151), list(map(lambda x: -x/5, range(151))), label='20% underestimation', c='red', linestyle='--')
    plt.title('Model residuals')
    plt.xlabel('True TTF values')
    plt.ylabel('Model error')
    plt.legend()
    if show:
        plt.show()


def residuals_zoom(trues, preds, max_TTF=30, show=True):
    plt.grid()
    plt.scatter(trues[trues <= max_TTF], preds[trues <= max_TTF]-trues[trues <= max_TTF], s=6)
    plt.plot(range(max_TTF+1), list(map(lambda x: x/5, range(max_TTF+1))), label='20% overestimation', c='darkred', linestyle='--')
    plt.plot(range(max_TTF+1), list(map(lambda x: -x/5, range(max_TTF+1))), label='20% underestimation', c='red', linestyle='--')
    plt.title('Model residuals at end of life')
    plt.xlabel('True TTF values')
    plt.ylabel('Model error')
    plt.legend()
    if show:
        plt.show()


def hist_residuals(trues, preds, show=True):
    plt.grid()
    plt.hist(preds-trues)
    plt.title('Residuals distribution')
    plt.xlabel('Model error')
    plt.ylabel('Count')
    if show:
        plt.show()


def residual_quadra_plot(trues, preds, show=True):
    fig = plt.figure()
    fig.add_subplot(221)
    plot_residuals(trues, preds, show=False)
    fig.add_subplot(222)
    hist_residuals(trues, preds, show=False)
    fig.add_subplot(223)
    residuals_zoom(trues, preds, max_TTF=30, show=False)
    fig.add_subplot(224)
    plot_error_repartition(trues, preds, show=False)
    if show:
        plt.show()


def plot_classification_report(trues, preds, time_to_failure, name=''):
    trues = pd.Series(trues).reset_index(drop=True)
    preds = pd.Series(preds).reset_index(drop=True)
    time_to_failure = pd.Series(time_to_failure).reset_index(drop=True)

    fig = plt.figure(figsize=(15, 8))
    grid = plt.GridSpec(2, 2)

    fig.add_subplot(grid[:, 0])
    heatmap(confusion_matrix(trues, preds), annot=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    fig.add_subplot(grid[0, 1])
    plt.hist(time_to_failure.loc[(trues == 1) & (preds == 0)])
    plt.title('False Negatives distribution')
    plt.xlabel('True time to failure')
    plt.ylabel('Count')

    fig.add_subplot(grid[1, 1])
    plt.hist(time_to_failure[(trues == 0) & (preds == 1)])
    plt.title('False Positives distribution')
    plt.xlabel('True time to failure')
    plt.ylabel('Count')

    plt.tight_layout()
    fig.suptitle('Classification report {}'.format(name))
    plt.show()
