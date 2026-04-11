import numpy as np
import pandas as pd
import pprint


def estimate_gbm(data, file, dt: float = 1/256):
    mu = np.mean(data)
    sigma_2 = np.var(data, ddof=1) / dt
    mu = mu / dt + sigma_2 / 2
    sigma = np.sqrt(sigma_2)

    result_formatted = {
        'parameters': {
            'mu': mu,
            'sigma': sigma
        },
    }

    pretty_string = pprint.pformat(result_formatted, indent=2)

    resultspath = file + "_gbm.txt"
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
        result = estimate_gbm(returns_list, file)

        print(result)


        _B = 1000
        params = {'mu': [0] * _B, 
                'sigma': [0] * _B}

        for r in range(_B):
            print(f"bootstrap {r}")
            bootstraps = np.random.choice(data, size=len(data), replace=True)

            result = estimate_gbm(np.array(bootstraps), file)

            for key in params.keys():
                params[key][r] = result['parameters'][key]

        with open("gbm_params.txt", 'a') as f:
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


    # data = pd.read_csv('less_vol_log_returns_3months.csv', dtype={'Log_Returns': float})['Log_Returns']


    # print('Mean drift:', mu)
    # print('Mean volatility:', sigma)

    # np_samples = np.array(data)
    # np_mu_hat = np.mean(np_samples)

    # _B = 1000
    # boot_np_mu_hats = [0] * _B
    # boot_np_sigma_hats = [0] * _B
    # for r in range(_B):
    #     bootstraps = np.random.choice(data, size=len(data), replace=True)

    #     mu = np.mean(bootstraps)
    #     sigma_2 = np.var(bootstraps, ddof=1) / dt
    #     mu = mu / dt + sigma_2/2
    #     sigma = np.sqrt(sigma_2)

    #     boot_np_mu_hats[r] = mu
    #     boot_np_sigma_hats[r] = sigma


    # boot_np_mu_hat_975, boot_np_mu_hat_25 = np.percentile(boot_np_mu_hats, [97.5, 2.5])
    # print(f"bootstrap mu: [{boot_np_mu_hat_25}, {boot_np_mu_hat_975}]")

    # boot_np_b_hat_975, boot_np_b_hat_25 = np.percentile(boot_np_sigma_hats, [97.5, 2.5])
    # print(f"bootstrap sigma: [{boot_np_b_hat_25}, {boot_np_b_hat_975}]")



# LESS VOLATILITY - 1 YEAR
# Mean drift: 0.1268483500241583
# Mean volatility: 0.2216388407804059
# bootstrap mu: [0.03653097855224578, 0.22357599643283163]
# bootstrap sigma: [0.2137843641768894, 0.2293736316852823]

# LESS VOLATILITY - 6 MONTHS
# Mean drift: 0.11497273080422699
# Mean volatility: 0.21622953579100895
# bootstrap mu: [-0.0033556703276934087, 0.2414145062588279]
# bootstrap sigma: [0.2044842291231765, 0.22772050771880173]

# LESS VOLATILITY - 3 MONTHS
# Mean drift: 0.24260556099467243
# Mean volatility: 0.23218320400155792
# bootstrap mu: [0.03536497793768283, 0.4401370250607015]
# bootstrap sigma: [0.21528802209885953, 0.25370977865218103]


# VOLATILITY - 1 YEAR
# Mean drift: 0.5156904187717846
# Mean volatility: 0.4027095127433022
# bootstrap mu: [0.33564332215389237, 0.6880526340705622]
# bootstrap sigma: [0.3849957346147754, 0.4197477480256986]

# VOLATILITY - 6 MONTHS
# Mean drift: 0.2966957891033525
# Mean volatility: 0.3791527039693616
# bootstrap mu: [0.08077127249862687, 0.5106813187201419]
# bootstrap sigma: [0.35764354823137573, 0.4025970099322128]

# VOLATILITY - 3 MONTHS
# Mean drift: 0.5525814981087294
# Mean volatility: 0.3927754953312561
# bootstrap mu: [0.2136448252657778, 0.8851312246858014]
# bootstrap sigma: [0.3584128162200525, 0.43114417472029115]



    # np.random.seed(1)
    
    # m = 1000
    # fig, ax = plt.subplots(3, 2)

    # # aapl: plot histogram of estimated rates
    # ax[0][0].hist(X_list[-1], bins=30, color='skyblue', edgecolor='black')

    # aapl_scipy_norm = Normal(mu = mus[-1], sigma = sigmas[-1])
    # x = np.linspace(np.min(X_list[-1]), np.max(X_list[-1]), m)
    # ax[0][0].plot(x, aapl_scipy_norm.pdf(x), color='black')

    # ax[0][0].grid(True)


    # ax[1][0].hist(X_list[0], bins=30, color='skyblue', edgecolor='black')

    # aapl_scipy_norm = Normal(mu = mus[0], sigma = sigmas[0])
    # x = np.linspace(np.min(X_list[0]), np.max(X_list[0]), m)
    # ax[1][0].plot(x, aapl_scipy_norm.pdf(x), color='black')

    # ax[1][0].grid(True)



    # ax[2][0].hist(X_list[1], bins=30, color='skyblue', edgecolor='black')

    # aapl_scipy_norm = Normal(mu = mus[1], sigma = sigmas[1])
    # x = np.linspace(np.min(X_list[1]), np.max(X_list[1]), m)
    # ax[2][0].plot(x, aapl_scipy_norm.pdf(x), color='black')

    # ax[2][0].grid(True)


    # ax[0][1].hist(X_list[2], bins=30, color='skyblue', edgecolor='black')

    # aapl_scipy_norm = Normal(mu = mus[2], sigma = sigmas[2])
    # x = np.linspace(np.min(X_list[2]), np.max(X_list[2]), m)
    # ax[0][1].plot(x, aapl_scipy_norm.pdf(x), color='black')

    # ax[0][1].grid(True)


    # ax[1][1].hist(X_list[3], bins=30, color='skyblue', edgecolor='black')

    # aapl_scipy_norm = Normal(mu = mus[3], sigma = sigmas[3])
    # x = np.linspace(np.min(X_list[3]), np.max(X_list[3]), m)
    # ax[1][1].plot(x, aapl_scipy_norm.pdf(x), color='black')

    # ax[1][1].grid(True)



    # ax[2][1].hist(X_list[4], bins=30, color='skyblue', edgecolor='black')

    # aapl_scipy_norm = Normal(mu = mus[4], sigma = sigmas[4])
    # x = np.linspace(np.min(X_list[4]), np.max(X_list[4]), m)
    # ax[2][1].plot(x, aapl_scipy_norm.pdf(x), color='black')

    # ax[2][1].grid(True)

    # ax[0][0].set_title('Stock 22')
    # ax[1][0].set_title('Stock 1')
    # ax[2][0].set_title('Stock 2')
    # ax[0][1].set_title('Stock 3')
    # ax[1][1].set_title('Stock 4')
    # ax[2][1].set_title('Stock 5')
    # fig.tight_layout()

    # plt.show()