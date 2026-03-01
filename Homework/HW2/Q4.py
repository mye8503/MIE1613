import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

import SimFunctions
import SimRNG
import SimClasses


from pprint import pprint 



if __name__ == "__main__":
    file_path_aapl = "AAPL.csv"
    file_path_sq = "SQ.csv"

    data_aapl = pd.read_csv(file_path_aapl)['Adj Close']
    data_sq = pd.read_csv(file_path_sq)['Adj Close']

    n = len(data_aapl)
    X_list_aapl = []
    for i in range(1, n):
        X_i = np.log(data_aapl[i] / data_aapl[i-1])
        X_list_aapl.append(X_i)

    # X_i = log(S(t_i) / S(t_(i-1))) ~ N(mu, sigma)
    # mu (mean) = 1/(n-1) * sum(X_i)
    # sigma^2 (variance) = 1/(n-1) * sum((X_i - mu)^2)
    aapl_mu = np.mean(X_list_aapl)
    # sample variance or regular?
    aapl_sigma_2 = np.var(X_list_aapl, ddof=1)

    print(aapl_mu, aapl_sigma_2)


    n = len(data_sq)
    X_list_sq = []
    for i in range(1, n):
        X_i = np.log(data_sq[i] / data_sq[i-1])
        X_list_sq.append(X_i)

    # X_i = log(S(t_i) / S(t_(i-1))) ~ N(mu, sigma)
    # mu (mean) = 1/(n-1) * sum(X_i)
    # sigma^2 (variance) = 1/(n-1) * sum((X_i - mu)^2)
    sq_mu = np.mean(X_list_sq)
    # sample variance or regular?
    sq_sigma_2 = np.var(X_list_sq, ddof=1)

    print(sq_mu, sq_sigma_2)


    # fix the seed of the random number generator
    np.random.seed(1)

    T = len(data_aapl)
    m = 1000
    mu = aapl_mu
    sigma = np.sqrt(aapl_sigma_2)
    InitialValue = data_aapl[0]
    dt = T / m

    times = np.linspace(0, T, m)
    X = InitialValue
    values = [X]

    for j in range(1,m):
        Z = np.random.normal(0,1)
        X = X * np.exp(mu * dt + sigma * np.sqrt(dt) * Z)
        values.append(X) 

    fig, ax = plt.subplots(1,2)
    ax[0].plot(times, values, color='r', label='GBM Fit')
    ax[0].plot(data_aapl, color='g', label='Historical Data')
    ax[0].set_title('AAPL Adj Close Price vs Fitted GBM')
    ax[0].set_xlabel('Time (t)')
    ax[0].set_ylabel('X(t)')
    ax[0].legend()


    # fix the seed of the random number generator
    np.random.seed(1)

    T = len(data_sq)
    m = 1000
    mu = sq_mu
    sigma = np.sqrt(sq_sigma_2)
    InitialValue = data_sq[0]
    dt = T / m

    times = np.linspace(0, T, m)
    X = InitialValue
    values = [X]

    for j in range(1,m):
        Z = np.random.normal(0,1)
        X = X * np.exp(mu * dt + sigma * np.sqrt(dt) * Z)
        values.append(X) 

    ax[1].plot(times, values, color='r', label='GBM Fit')
    ax[1].plot(data_sq, color='g', label='Historical Data')
    ax[1].set_title('SQ Adj Close Price vs Fitted GBM')
    ax[1].set_xlabel('Time (t)')
    ax[1].set_ylabel('X(t)')
    ax[1].legend()
    plt.show()


