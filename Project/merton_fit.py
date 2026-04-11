import numpy as np
import pandas as pd
import math
import pprint
from scipy.optimize import minimize
from scipy.stats import norm


def log_likelihood(params, returns, sigma: float, dt: float = 1/256) -> float:
    mu, lamb, nu, omega = params

    # check bounds
    if lamb <= 0 or omega <= 0:
        return -np.inf
    
    # adjust paper's reported values to fit equation
    mu_daily = mu * dt
    sigma_daily = sigma * np.sqrt(dt)

    # jump size adjustment
    k = np.exp(nu + omega**2/2) - 1

    n_max = 10
    log_lik = 0.0
    for r in returns:
        weight_sum = 0.0

        for n in range(n_max + 1):
            # poisson pmf for n jumps
            p_n = np.exp(-lamb * dt) * (lamb * dt)**n / math.factorial(n)

            # negligible p_n
            if p_n <= 1e-12:
                continue
            
            # values for gaussian pdf
            mu_n = (mu_daily - 0.5 * sigma_daily**2 - lamb * k) + n * nu

            sigma2_n = sigma_daily**2 + n * omega**2

            # invalid value
            if sigma2_n <= 0:
                continue

            weight_sum += p_n * norm.pdf(r, mu_n, np.sqrt(sigma2_n))

        if weight_sum > 0:
            log_lik += np.log(weight_sum)

    return log_lik


def estimate_merton(returns: np.ndarray, sigma: float, file) -> dict:
    # Initial guess
    # [mu, lamb, nu, omega]
    initial_params = np.array([0.1164, 0.7114, -0.0081, 0.0786])

    
    def objective(params):
        return -log_likelihood(params, returns, sigma)
    

    # Run optimization
    result = minimize(
        objective, 
        initial_params, 
        method='L-BFGS-B',
        options={'maxiter': 1000},
    )

    result_formatted = {
        'parameters': {
            'mu': result.x[0],
            'lamb': result.x[1],
            'nu': result.x[2],
            'omega': result.x[3]
        },
        'success': result.success,
        'message': result.message
    }

    pretty_string = pprint.pformat(result_formatted, indent=2)

    resultspath = file + "_merton.txt"
    with open(resultspath, 'a') as f:
        f.write(pretty_string)
        f.write("\n\n")

    return result_formatted
     


if __name__ == "__main__":
    files = ["less_vol_log_returns", "less_vol_log_returns_6months", "less_vol_log_returns_3months", "vol_log_returns", "vol_log_returns_6months", "vol_log_returns_3months"]

    for file in files:
        filename = file + ".csv"
        data = pd.read_csv(filename, dtype={'Log_Returns': float})['Log_Returns']

        # sigma not included in paper MLE 
        # assume sigma = sqrt(var(log_returns) / dt)
        dt = 1/256 
        sigma = np.sqrt(np.var(data, ddof=1) / dt)

        print(sigma)

        returns_list = np.array(data)
        result = estimate_merton(returns_list, sigma, file)

        print(result)


        _B = 50
        params = {'mu': [0] * _B, 
                'lamb': [0] * _B, 
                'nu': [0] * _B, 
                'omega': [0] * _B}


        for r in range(_B):
            print(f"bootstrap {r}")
            bootstraps = np.random.choice(data, size=len(data), replace=True)

            result = estimate_merton(np.array(bootstraps), sigma, file)

            for key in params.keys():
                params[key][r] = result['parameters'][key]


        with open("merton_params.txt", 'a') as f:
            f.write("=======================\n")
            f.write(f"Results for {file}\n")
            for key, val in params.items():
                boot_975, boot_25 = np.percentile(val, [97.5, 2.5])
                f.write(f"{key}: {np.mean(val)} [{boot_975}, {boot_25}]\n")
            f.write("=======================\n\n")

            print("=======================\n")
            print(f"Results for {file}\n")
            for key, val in params.items():
                boot_975, boot_25 = np.percentile(val, [97.5, 2.5])
                print(f"{key}: {np.mean(val)} [{boot_975}, {boot_25}]\n")
            print("=======================\n\n")


# LESS VOLATILE - 1 YEAR - 100 BOOTSTRAP SAMPLES
# bootstrap mu: [0.33773518174453726, 0.03309838142174046]
# bootstrap lamb: [8.641789697028212, 1.5230975407325467]
# bootstrap nu: [-0.0005339194991836867, -0.0030061262028769978]
# bootstrap omega: [0.07382143976158659, 0.03473795170114579]