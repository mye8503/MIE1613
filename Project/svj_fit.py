import numpy as np
import pandas as pd
import pprint
from scipy.stats import norm
from scipy.optimize import minimize


class SVJParticleFilter:   
    def __init__(self, n_particles = 5000, resample_threshold = 0.5, seed = -1):
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold  

        # # use CRN to get the same log_likelihood from evaluating the same parameters
        # if seed > 0:
        #     np.random.seed(seed)


    def log_likelihood(self, params, returns, dt = 1/250):
        # [kappa, theta, xi, rho, v0, mu, lamb, mu_J, sigma_J]
        kappa, theta, xi, rho, v0, mu, lamb, mu_J, sigma_J = params
        
        # check constraints
        # feller condition
        if 2 * kappa * theta <= xi**2:
            return -np.inf
        
        # jump intensity must be between 0 and 1
        if lamb <= 0 or lamb >= 1:
            return -np.inf
        
        # positive jump volatility
        if sigma_J <= 0:
            return -np.inf
        
        # parameter bounds for numerical stability
        if kappa <= 0 or theta <= 0 or xi <= 0 or v0 <= 0:
            return -np.inf
        
        jump_prob = lamb * dt  # probability of jump in each time step
        
        # initialize particles
        particles_v = np.ones(self.n_particles) * v0
        particles_v = np.maximum(particles_v, 1e-8)
        weights = np.ones(self.n_particles) / self.n_particles
        
        log_lik = 0.0
        
        for r in returns:
            # update volatility particles
            # correlated random variables - needs CRN to be more precise
            z1 = np.random.normal(0, 1, self.n_particles)
            z2 = np.random.normal(0, 1, self.n_particles)
            z2 = rho * z1 + np.sqrt(1 - rho**2) * z2
            
            # euler-maruyama to update volatility
            drift = kappa * (theta - particles_v) * dt
            diffusion = xi * np.sqrt(particles_v * dt) * z2
            particles_v_new = particles_v + drift + diffusion
            particles_v_new = np.maximum(particles_v_new, 1e-8)
            
            # expected return given volatility

            # drift term
            drift_term = (mu - 0.5 * particles_v_new) * dt
            
            # no jump: q_t = 0, prob = 1 - lamb*dt
            cond_mean_no_jump = drift_term
            cond_var_no_jump = particles_v_new * dt
            cond_std_no_jump = np.sqrt(cond_var_no_jump)
            
            lik_no_jump = norm.pdf(r, loc=cond_mean_no_jump, scale=cond_std_no_jump)
            
            # one jump: q_t = 1, prob = lamb*dt
            cond_mean_jump = drift_term + mu_J
            cond_var_jump = particles_v_new * dt + sigma_J**2
            cond_std_jump = np.sqrt(cond_var_jump)
            
            lik_jump = norm.pdf(r, loc=cond_mean_jump, scale=cond_std_jump)
            
            # now combine
            likelihoods = (1 - jump_prob) * lik_no_jump + jump_prob * lik_jump
            
            # update particle weights
            unnormalized_weights = weights * likelihoods
            weight_sum = np.sum(unnormalized_weights)
            
            if weight_sum == 0:
                return -np.inf

            # now normalize
            weights = unnormalized_weights / weight_sum
            
            # likelihood contribution
            log_lik += np.log(weight_sum)
            
            # resample weights if needed
            n_eff = 1.0 / np.sum(weights**2)
            if n_eff < self.resample_threshold * self.n_particles:
                indices = np.random.choice(
                    self.n_particles, 
                    size=self.n_particles, 
                    p=weights,
                    replace=True
                )
                particles_v = particles_v_new[indices]
                weights = np.ones(self.n_particles) / self.n_particles

            # update particles regardless of resampling
            else:
                particles_v = particles_v_new
        
        return log_lik


def estimate_svj_pf(returns, file, n_particles = 5000):
    pf = SVJParticleFilter(n_particles = n_particles, seed=123)

    # initial guess - paper values for less-vol-1-year
    # [kappa, theta, xi, rho, v0, mu, lamb, mu_J, sigma_J]
    initial_params = np.array([1.6549, 0.1557, 0.2806, 0.0257, 0.0336, 0.1055, 0.6263, -0.0062, 0.2483])
    
    def objective(params):
        return -pf.log_likelihood(params, returns)
    
    # run optimization
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=[
            (0.01, 20),     # kappa
            (0.001, 1),     # theta
            (0.01, 2),      # xi
            (-0.999, 0.999),# rho
            (1e-5, 1),      # v0
            (-1, 1),        # mu
            (1e-4, 249),    # lamb
            (-1, 1),        # mu_J
            (1e-4, 2),      # sigma_J
        ],
        # method='Powell',
        options={
            'maxiter': 1000, 
            # 'xtol': 0.01,      
            'ftol': 0.1,      
            }
    )

    result_formatted = {
        'parameters': {
            'kappa': result.x[0],
            'theta': result.x[1],
            'xi': result.x[2],
            'rho': result.x[3],
            'v0': result.x[4],
            'mu': result.x[5],
            'lamb': result.x[6],
            'mu_J': result.x[7],
            'sigma_J': result.x[8]
        },
        'success': result.success,
        'message': result.message
    }

    pretty_string = pprint.pformat(result_formatted, indent=2)

    resultspath = file + "_svj.txt"
    with open(resultspath, 'a') as f:
        f.write(pretty_string)
        f.write("\n\n")

    return result_formatted


if __name__ == "__main__":
    files = ["less_vol_log_returns", "less_vol_log_returns_6months", "less_vol_log_returns_3months", 
             "vol_log_returns", "vol_log_returns_6months", "vol_log_returns_3months", 
             "new_data_log_returns", "new_data_log_returns_6months", "new_data_log_returns_3months"]

    for file in files:
        filename = file + ".csv"

        data = pd.read_csv(f"data/{filename}", dtype={'Log_Returns': float})['Log_Returns']

        returns_list = np.array(data)
        result = estimate_svj_pf(returns_list, file)

        print(result)

        _B = 50
        params = {'kappa': [0] * _B, 
                'theta': [0] * _B, 
                'xi': [0] * _B, 
                'rho': [0] * _B, 
                'v0': [0] * _B, 
                'mu': [0] * _B,
                'lamb': [0] * _B,
                'mu_J': [0] * _B,
                'sigma_J': [0] * _B}


        for r in range(_B):
            print(f"bootstrap {r}")
            bootstraps = np.random.choice(data, size=len(data), replace=True)

            result = estimate_svj_pf(np.array(bootstraps), file)

            for key in params.keys():
                params[key][r] = result['parameters'][key]


        with open("svj_params.txt", 'a') as f:
            f.write("=======================\n")
            f.write(f"Results for {file}\n")
            for key, val in params.items():
                boot_25, boot_975 = np.percentile(val, [2.5, 97.5])
                f.write(f"{key}: {np.mean(val)} [{boot_25}, {boot_975}]\n")
            f.write("=======================\n\n")

            print("=======================\n")
            print(f"Results for {file}\n")
            for key, val in params.items():
                boot_25, boot_975 = np.percentile(val, [2.5, 97.5])
                print(f"{key}: {np.mean(val)} [{boot_25}, {boot_975}]\n")
            print("=======================\n\n")