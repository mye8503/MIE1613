
import numpy as np

# ========== PART A ========== #
# F(X) = 1/2 * e^((x - mu) / b))        x < mu
# F(X) = 1 - 1/2 * e^(-(x - mu) / b)    x >= mu
# 
# inversion method: X = F^-1(U)
#
# F^-1(X) = b * ln(2x) + mu             x < 1/2
# F^-1(X) = -b * ln(2(1 - x)) + mu       x >= 1/2

if __name__ == "__main__":
    # ========== PART C ========== #
    n = 1000
    mu = 0
    b = 2

    # fix random number seed
    np.random.seed(123)

    # ASK: np quartiles are diff from below method
    samples = [0] * n
    for i in range(n):
        U = np.random.uniform()

        if U < 0.5: samples[i] = b * np.log(2 * U) + mu
        else: samples[i] = -b * np.log(2 * (1 - U)) + mu

    sorted_samples = sorted(samples)
    # account for zero indexing
    mu_hat = (sorted_samples[n // 2 - 1] + sorted_samples[n // 2]) / 2
    q_25 = (sorted_samples[n // 4 - 1] + sorted_samples[n // 4]) / 2
    q_75 = (sorted_samples[n // 4 * 3 - 1] + sorted_samples[n // 4 * 3]) / 2
    b_hat = (q_75 - q_25) / (2 * np.log(2))

    print(f"mu_hat: {mu_hat}  b_hat: {b_hat}")


    # mu_hat: -0.03395416519832477  b_hat: 1.8531994091555655
    # compare to mu = 0, b = 2


    # ========== PART D ========== #
    # step 1: For r = 1, . . . , B, draw a bootstrap sample
    # by sampling with replacement from the original data.
    np_samples = np.array(samples)

    np_mu_hat = np.median(np_samples)
    np_75, np_25 = np.percentile(np_samples, [75, 25])
    np_b_hat = (np_75 - np_25) / (2 * np.log(2))

    print(f"np_mu_hat: {np_mu_hat}  np_b_hat: {np_b_hat}")

    _B = 1000
    boot_np_mu_hats = [0] * _B
    boot_np_b_hats = [0] * _B
    for r in range(_B):
        bootstraps = np.random.choice(np_samples, size=len(np_samples), replace=True)

        # step 2: Compute bootstrap estimates using the same 
        # matching formulas as in part (c).
        boot_np_mu_hats[r] = np.median(bootstraps)
        boot_np_75, boot_np_25 = np.percentile(bootstraps, [75, 25])
        boot_np_b_hats[r] = (boot_np_75 - boot_np_25) / (2 * np.log(2))

    # step 3: Let µˆ∗(0.025) and µˆ*(0.975) be the 0.25th and 
    # 0.975th quantiles of {µˆ∗(r)}B_r=1. Report the 
    # CI [ˆµ∗(0.025), µˆ∗(0.975)]. Do the same for b.

    boot_np_mu_hat_975, boot_np_mu_hat_25 = np.percentile(boot_np_mu_hats, [97.5, 2.5])
    print(f"bootstrap mu: [{boot_np_mu_hat_25}, {boot_np_mu_hat_975}]")

    boot_np_b_hat_975, boot_np_b_hat_25 = np.percentile(boot_np_b_hats, [97.5, 2.5])
    print(f"bootstrap b: [{boot_np_b_hat_25}, {boot_np_b_hat_975}]")

    # bootstrap mu: [-0.17189553684683748, 0.08059304818729546] --> mu = 0 falls inside this interval
    # bootstrap b: [1.6750351883537922, 2.037993017661069] --> b = 2 falls inside this interval

    