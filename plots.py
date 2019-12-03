import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from collections import OrderedDict

from statsmodels.graphics.correlation import plot_corr
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def plot_engine_signals(engine_id, df, signals, scale_signals=True):
    df = df.copy()
    if scale_signals:
        scaler = MinMaxScaler()
        df.loc[:, signals] = scaler.fit_transform(df.loc[:, signals])

    if 'time_to_failure' in df.columns:
        [plt.plot(df.loc[engine_id, "time_to_failure"],
                  df.loc[engine_id, signal]) for signal in signals]
        plt.xlabel('Remaining Useful Life')
        plt.xlim(0, 200)
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


def plot_superposed_engine_signals(engine_ids, df, signals):
    for signal in signals:
        [plt.plot(df.loc[engine_id, "time_to_failure"], df.loc[engine_id, signal], linewidth=.4)
         for engine_id in engine_ids]
        plt.xlabel('Remaining Useful Life')
        plt.ylabel(signal)
        plt.xlim(0, 200)
        ax = plt.gca()
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.title("Evolution of {signal} for {number_of_engines} engines".format(signal=signal,
                                                                                 number_of_engines=len(engine_ids)))
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


# Plotting time !
# Let's see how signals evolve during the lifetime of these engines
def plot_3d_lifetime_paths(train):
    # PC1 signals seem to gather at 1 for end of lifetime
    # and PC2 signals gather somewhere around 0.5 for beginning of life.
    # Furthermore, the data points are more spaced out at end of life (derivatives are higher!)
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


def plot_error_repartition(y_true, y_pred):
    errors = OrderedDict.fromkeys(sorted(set(y_true)))
    for i in set(y_true):
        errors[i] = []
    [errors[true].append(np.abs(pred-true)) for true, pred in zip(y_true, y_pred)]
    plt.plot(list(errors.keys()), list(map(np.mean, errors.values())))
    plt.title("Error repartition")
    plt.show()
