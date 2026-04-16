import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from params import GBM_params, Heston_params, Merton_params, SVJ_params
from paper_params import GBM_paparams, Heston_paparams, Merton_paparams, SVJ_paparams


class GBM:
    def __init__(self, params):
        self.mu, self.sigma = params

    def predict(self, x0, num_days, dt=1/256):
        # use CRN across models
        # np.random.seed(123)
        
        X = x0
        values = [X]
        # first day already included
        for _ in range(1, num_days):
            Z = np.random.normal(0,1)
            X = X * np.exp(self.mu * dt + self.sigma * np.sqrt(dt) * Z)
            values.append(X)

        return values
    

class Heston:
    def __init__(self, params):
        self.kappa, self.theta, self.xi, self.rho, self.v0, self.mu = params

    def predict(self, x0, num_days, dt=1/256):
        # use CRN across models
        # np.random.seed(123)
        
        X = x0
        v = self.v0
        values = [X]
        # first day already included
        for _ in range(1, num_days):
            # stock BM
            Z1 = np.random.normal(0,1)
            # volatility BM
            Z2 = np.random.normal(0,1)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

            # update variance
            v = np.abs(v + self.kappa * (self.theta - v) * dt + self.xi * np.sqrt(v) * Z2 * np.sqrt(dt))
            # update price
            X = X * np.exp((self.mu - 0.5 * v) * dt + np.sqrt(v) * Z1 * np.sqrt(dt))

            values.append(X)

        return values
    

class Merton:
    def __init__(self, params):
        self.mu, self.lamb, self.nu, self.omega, self.sigma = params

    def predict(self, x0, num_days, dt=1/256):
        # use CRN across models
        # np.random.seed(123)
        
        X = x0
        values = [X]
        # first day already included
        for _ in range(1, num_days):
            # stock BM
            Z1 = np.random.normal(0,1)
            # jumps
            N = np.random.poisson(self.lamb * dt)
            jumps = np.sum(np.random.normal(self.nu, self.omega, N))

            # update price
            X = X * np.exp((self.mu - 0.5 * self.omega**2 - self.lamb * self.nu) * dt
                           + self.omega * np.sqrt(dt) * Z1 + jumps)

            values.append(X)

        return values
    

class SVJ:
    def __init__(self, params):
        self.kappa, self.theta, self.xi, self.rho, self.v0, self.mu, self.lamb, self.mu_J, self.sigma_J = params

    def predict(self, x0, num_days, dt=1/256):
        # use CRN across models
        # np.random.seed(123)
        
        X = x0
        v = self.v0
        values = [X]
        # first day already included
        for _ in range(1, num_days):
            # stock BM
            Z1 = np.random.normal(0,1)
            # volatility BM
            Z2 = np.random.normal(0,1)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

            # jumps
            N = np.random.poisson(self.lamb * dt)
            jumps = np.sum(np.random.normal(self.mu_J, self.sigma_J, N))

            # update variance
            v = np.abs(v + self.kappa * (self.theta - v) * dt + self.xi * np.sqrt(v) * Z2 * np.sqrt(dt))
            # update price
            X = X * np.exp((self.mu - 0.5 * v - self.lamb * self.mu_J) * dt + np.sqrt(v) * Z1 * np.sqrt(dt) + jumps)

            values.append(X)

        return values
    

def mean_squared_error(true_values, predicted_values):
    SE = 0
    for true, predicted in zip(true_values, predicted_values):
        SE += (true - predicted) ** 2

    return SE / len(true_values)



if __name__ == '__main__':
    files = ["less_vol_interpolated", "less_vol_interpolated_6months", "less_vol_interpolated_3months", "vol_interpolated", "vol_interpolated_6months", "vol_interpolated_3months", "new_data", "new_data_6months", "new_data_3months"]

    names = ["GBM", "Heston", "Merton", "SVJ"]

    for filename in files:
        print(filename)
        if "new" in filename:
            num_stocks = 33
        else:
            num_stocks = 22

        data = pd.read_csv(f"data/{filename}.csv")

        gbm_model = GBM(GBM_params[filename])
        # heston_model = Heston(Heston_params[filename])
        # merton_model = Merton(Merton_params[filename])
        # svj_model = SVJ(SVJ_params[filename])

        # gbm_model = GBM(GBM_paparams[filename])
        # heston_model = Heston(Heston_paparams[filename])
        # merton_model = Merton(Merton_paparams[filename])
        # svj_model = SVJ(SVJ_paparams[filename])

        gbm_MSE = 0
        heston_MSE = 0
        merton_MSE = 0
        svj_MSE = 0
        for stock in range(num_stocks):
            if stock > 0:
                d = data[f"Close.{stock}"]
            else:
                d = data["Close"]

            data_results = d.iloc[3:].astype(float)

            N = 10000
            for _ in range(N):
                gbm_results = gbm_model.predict(float(d.iloc[3]), len(d))
                # heston_results = heston_model.predict(float(d.iloc[3]), len(d))
                # merton_results = merton_model.predict(float(d.iloc[3]), len(d))
                # svj_results = svj_model.predict(float(d.iloc[3]), len(d))

                gbm_MSE += mean_squared_error(data_results, gbm_results)
                # heston_MSE += mean_squared_error(data_results, heston_results)
                # merton_MSE += mean_squared_error(data_results, merton_results)
                # svj_MSE += mean_squared_error(data_results, svj_results)


        gbm_RMSE = np.sqrt(gbm_MSE / num_stocks / N)
        # heston_RMSE = np.sqrt(heston_MSE / num_stocks / N)
        # merton_RMSE = np.sqrt(merton_MSE / num_stocks / N)
        # svj_RMSE = np.sqrt(svj_MSE / num_stocks / N)

        print("GBM", filename, gbm_RMSE)
        # print("Heston", filename, heston_RMSE)
        # print("Merton", filename, merton_RMSE)
        # print("SVJ", filename, svj_RMSE)
        print()



