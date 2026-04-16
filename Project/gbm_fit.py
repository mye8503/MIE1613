import numpy as np
import pandas as pd
import pprint


def estimate_gbm(data, file, dt: float = 1/250):
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

    # resultspath = file + "_gbm.txt"
    # with open(resultspath, 'a') as f:
    #     f.write(pretty_string)
    #     f.write("\n\n")

    return result_formatted



if __name__ == "__main__":
    files = ["less_vol_log_returns", "less_vol_log_returns_6months", "less_vol_log_returns_3months", "vol_log_returns", "vol_log_returns_6months", "vol_log_returns_3months"]
    new_files = ["new_data_log_returns", "new_data_log_returns_6months", "new_data_log_returns_3months"]

    for file in new_files:
        filename = file + ".csv"
        data = pd.read_csv(f"data\{filename}", dtype={'Log_Returns': float})['Log_Returns']

        returns_list = np.array(data)
        result = estimate_gbm(returns_list, file)

        print(result)


        _B = 1000
        params = {'mu': [0] * _B, 
                'sigma': [0] * _B}

        for r in range(_B):
            # print(f"bootstrap {r}")
            bootstraps = np.random.choice(data, size=len(data), replace=True)

            result = estimate_gbm(np.array(bootstraps), file)

            for key in params.keys():
                params[key][r] = result['parameters'][key]

        # with open("gbm_params.txt", 'a') as f:
        #     f.write("=======================\n")
        #     f.write(f"Results for {file}\n")
        #     for key, val in params.items():
        #         boot_975, boot_25 = np.percentile(val, [97.5, 2.5])
        #         f.write(f"{key}: {np.mean(val)} [{boot_975}, {boot_25}]\n")
        #     f.write("=======================\n\n")

        print("=======================\n")
        print(f"Results for {file}\n")
        for key, val in params.items():
            boot_975, boot_25 = np.percentile(val, [97.5, 2.5])
            print(f"{key}: {np.mean(val)} [{boot_975}, {boot_25}]\n")
        print("=======================\n\n")



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
