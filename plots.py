import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from collections import OrderedDict
import yellowbrick

from statsmodels.graphics.correlation import plot_corr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_engine_signals(engine_id, df, signals, scale_signals=True):
    df = df.copy()
    if scale_signals:
        scaler = StandardScaler()
        df.loc[:, signals] = scaler.fit_transform(df.loc[:, signals])

    if 'time_to_failure' in df.columns:
        [plt.plot(df.loc[engine_id, "time_to_failure"],
                  df.loc[engine_id, signal]) for signal in signals]
        plt.xlabel('Remaining Useful Life')
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
    for signal in signals:
        [plt.plot(df.loc[engine_id, "time_to_failure"], df.loc[engine_id, signal], linewidth=.4)
         for engine_id in engine_ids]
        plt.xlabel('Remaining Useful Life')
        plt.ylabel(signal)
        plt.xlim(0, df.time_to_failure.max())
        ax = plt.gca()
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.title("Evolution of {signal} for {number_of_engines} engines".format(signal=signal,
                                                                                 number_of_engines=len(engine_ids)))
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
    plt.xticks(list(range(1, 10)))
    plt.ylim(y[1], 1)
    plt.xlim(1, len(cols))
    plt.grid()
    plt.show()


def plot_correlation_map(train, sensor_columns):
    corr = train[sensor_columns].corr()
    plot_corr(corr, xnames=sensor_columns, ynames=sensor_columns, title='Sensor Correlations')
    plt.show()


def plot_3d_lifetime_paths(train):
    # PC1 signals seem to gather at 1 for end of lifetime
    # and PC2 signals gather somewhere around 0.5 for beginning of life.
    # Furthermore, the data points are more spaced out at end of life (derivatives are higher)
    # We might want to add this information to the dataset by computing derivatives or variance
    # All of this will give the model insight about health conditions.
    if 'PC1' not in train.columns:
        print('Please perform PCA with 2 principal components before calling plot_3d_lifetime_paths')
        return
    fig = plt.figure().gca(projection='3d')
    fig.scatter(train.PC1,
                train.PC2,
                train.time_to_failure,
                s=0.02,
                c=train.time_to_failure,
                cmap='magma')
    fig.set_xlabel('PC1')
    fig.set_ylabel('PC2')
    fig.set_zlabel('time_to_failure')
    fig.set_zlim3d(0, 250)
    fig.set_title("Engines lifetime paths")
    plt.show()


def plot_error_repartition(trues, preds, show=True, model_name="", save=False):
    errors = OrderedDict.fromkeys(sorted(set(trues)))
    for i in set(trues):
        errors[i] = []
    [errors[true].append(np.abs(pred-true)) for true, pred in zip(trues, preds)]
    plt.plot(list(errors.keys()), list(map(np.mean, errors.values())), label='Mean error')
    plt.plot(list(errors.keys()), list(map(np.max, errors.values())), label='Max error')
    plt.plot(list(errors.keys()), list(map(np.median, errors.values())), label='Median error')
    plt.xlabel('True RUL values')
    plt.ylabel('Absolute error')
    plt.legend()
    plt.title("Absolute error repartition for {model_name}".format(model_name=model_name))
    if save:
        plt.savefig("regression_results_and_plots/Absolute error repartition for {model_name}.png".format(model_name=model_name))
    if show:
        plt.show()


def plot_residuals(trues, preds, show=True):
    plt.grid()
    plt.scatter(trues, preds-trues, s=3)
    plt.plot(range(151), list(map(lambda x: x/5, range(151))), label='20% overestimation', c='darkred', linestyle='--')
    plt.plot(range(151), list(map(lambda x: -x/5, range(151))), label='20% underestimation', c='red', linestyle='--')
    plt.title('Model residuals')
    plt.xlabel('True RUL values')
    plt.ylabel('Model error')
    plt.legend()
    plt.grid()
    if show:
        plt.show()


def residuals_zoom(trues, preds, max_rul=30, show=True):
    plt.grid()
    plt.scatter(trues[trues <= max_rul], preds[trues <= max_rul]-trues[trues <= max_rul], s=6)
    plt.plot(range(max_rul+1), list(map(lambda x: x/5, range(max_rul+1))), label='20% overestimation', c='darkred', linestyle='--')
    plt.plot(range(max_rul+1), list(map(lambda x: -x/5, range(max_rul+1))), label='20% underestimation', c='red', linestyle='--')
    plt.title('Model residuals at end of life')
    plt.xlabel('True RUL values')
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
    residuals_zoom(trues, preds, max_rul=30, show=False)
    fig.add_subplot(224)
    plot_error_repartition(trues, preds, show=False)
    if show:
        plt.show()
