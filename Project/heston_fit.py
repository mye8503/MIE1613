import numpy as np
import pandas as pd
import pprint
from scipy.stats import norm, gamma
from scipy.optimize import minimize


class HestonParticleFilter:
    def __init__(self, n_particles: int = 5000, resample_threshold: float = 0.5, seed: int = -1):
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold

        # use CRN to get the same log_likelihood from evaluating the same parameters
        if seed > 0:
            np.random.seed(seed)
        
    def log_likelihood(self, params: np.ndarray, returns: np.ndarray, dt: float = 1/256) -> float:
        """
        Compute log-likelihood using particle filter
        
        params: [kappa, theta, xi, rho, v0, mu]
        """
        kappa, theta, xi, rho, v0, mu = params
        
        # constraints: feller condition
        if 2*kappa*theta <= xi**2:
            return -np.inf
        
        T = len(returns)
        
        # Initialize particles
        particles_v = np.ones(self.n_particles) * v0
        particles_v = np.maximum(particles_v, 1e-8)
        weights = np.ones(self.n_particles) / self.n_particles
        
        log_lik = 0.0
        
        for r in returns:
            # Step 1: Predict - evolve volatility particles
            # Generate correlated shocks

            # needs CRN
            eps1 = np.random.normal(0, 1, self.n_particles)
            eps2 = np.random.normal(0, 1, self.n_particles)
            eps2 = rho * eps1 + np.sqrt(1 - rho**2) * eps2
            
            # Euler-Maruyama for volatility
            drift = kappa * (theta - particles_v) * dt
            diffusion = xi * np.sqrt(particles_v * dt) * eps2
            particles_v_new = particles_v + drift + diffusion
            particles_v_new = np.maximum(particles_v_new, 1e-8)
            
            # Step 2: Update - compute likelihood of observed return
            # Expected return given volatility
            cond_mean = (mu - 0.5 * particles_v_new) * dt
            cond_var = particles_v_new * dt
            cond_std = np.sqrt(cond_var)
            
            # Likelihood of this return for each particle
            likelihoods = norm.pdf(r, loc=cond_mean, scale=cond_std)
            
            # Update weights
            unnormalized_weights = weights * likelihoods
            weight_sum = np.sum(unnormalized_weights)
            
            if weight_sum == 0:
                # Numerical underflow - return poor likelihood
                return -np.inf
            
            normalized_weights = unnormalized_weights / weight_sum
            
            # Step 3: Likelihood contribution
            log_lik += np.log(weight_sum / self.n_particles)
            
            # Step 4: Resample if needed
            n_eff = 1.0 / np.sum(normalized_weights**2)
            if n_eff < self.resample_threshold * self.n_particles:
                indices = np.random.choice(
                    self.n_particles, 
                    size=self.n_particles, 
                    p=normalized_weights,
                    replace=True
                )
                particles_v_new = particles_v_new[indices]
                normalized_weights = np.ones(self.n_particles) / self.n_particles
            
            # Update for next iteration
            particles_v = particles_v_new
            weights = normalized_weights
            
            # Optional: early stopping if likelihood explodes
            if np.isnan(log_lik) or np.isinf(log_lik):
                return -np.inf
        
        return log_lik
    

def estimate_heston_pf(returns: np.ndarray, file, n_particles: int = 5000) -> dict:
    """
    Estimate Heston parameters using particle filter MLE
    """
    pf = HestonParticleFilter(n_particles=n_particles)
    
    # initial guess - paper values
    # [kappa, theta, xi, rho, v0, mu]
    initial_params = np.array([1.5068, 0.1543, 0.2871, -0.0932, 0.0337, 0.1164])
    

    def objective(params):
        return -pf.log_likelihood(params, returns)
    

    # Run optimization
    result = minimize(
        objective, 
        initial_params, 
        method='L-BFGS-B',
        options={'maxiter': 1000},
    )

    result_formatted = {
        'parameters': {
            'kappa': result.x[0],
            'theta': result.x[1],
            'xi': result.x[2],
            'rho': result.x[3],
            'v0': result.x[4],
            'mu': result.x[5]
        },
        'success': result.success,
        'message': result.message
    }

    pretty_string = pprint.pformat(result_formatted, indent=2)

    resultspath = file + "_heston.txt"
    with open(resultspath, 'a') as f:
        f.write(pretty_string)
        f.write("\n\n")

    return result_formatted


if __name__ == "__main__":
    files = ["less_vol_log_returns", "less_vol_log_returns_6months", "less_vol_log_returns_3months", "vol_log_returns", "vol_log_returns_6months", "vol_log_returns_3months"]

    for file in files:
        filename = file + ".csv"
        data = pd.read_csv(filename, dtype={'Log_Returns': float})['Log_Returns']

        returns_list = np.array(data)
        result = estimate_heston_pf(returns_list, file)

        print(result)


        _B = 50
        params = {'kappa': [0] * _B, 
                'theta': [0] * _B, 
                'xi': [0] * _B, 
                'rho': [0] * _B, 
                'v0': [0] * _B, 
                'mu': [0] * _B}


        for r in range(_B):
            print(f"bootstrap {r}")
            bootstraps = np.random.choice(data, size=len(data), replace=True)

            result = estimate_heston_pf(np.array(bootstraps), file)

            for key in params.keys():
                params[key][r] = result['parameters'][key]


        with open("heston_params.txt", 'a') as f:
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



# bootstrap kappa: [1.50000047296594, 1.4999993836372636]
# bootstrap theta: [0.1500017026738224, 0.14999993062866074]
# bootstrap xi: [0.30000110469510016, 0.29999811546615807]
# bootstrap rho: [-0.0499990957638866, -0.05000169707225707]
# bootstrap v0: [0.04000164774535792, 0.0399997664773196]
# bootstrap mu: [0.15000027636697552, 0.14999858600471883]


# LESS VOLATILE - 1 YEAR - 100 BOOTSTRAP SAMPLES
# bootstrap kappa: [1.506802012904741, 1.506799425246059]
# bootstrap theta: [0.1543027471443737, 0.1542999142725817]
# bootstrap xi: [0.28710178987835283, 0.28709909565359976]
# bootstrap rho: [-0.0931982458713911, -0.09320052554381869]
# bootstrap v0: [0.033701585374156205, 0.03369849333358351]
# bootstrap mu: [0.11640193480478551, 0.11639842993255]