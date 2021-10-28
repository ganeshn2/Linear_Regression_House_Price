import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.mstats import normaltest
from scipy.stats import boxcox


def plot_square_normal_data():
    data = np.square(np.random.normal(loc=5, size=1000))
    plt.hist(data)
    plt.show()
    return data


def plot_exponential_data():
    data = np.exp(np.random.normal(size=1000))
    plt.hist(data)
    plt.show()
    return data


def normality_test(X):
    log_X = np.log(X)
    log_value = normaltest(log_X)
    sqrt_X = np.sqrt(X)
    sqrt_value = normaltest(sqrt_X)
    bc_X = boxcox(X)
    bc_price = bc_X[0]
    bc_value = normaltest(bc_price)
    return log_value,sqrt_value,bc_value


def boxcox_test(X):
    bc_X = boxcox(X)
    bc_price = bc_X[0]
    bc_lam = bc_X[1]
    return bc_price, bc_lam










