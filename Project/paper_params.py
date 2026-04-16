# mu, sigma
GBM_paparams = {
    "less_vol_interpolated": [0.1164, 0.1797],
    "less_vol_interpolated_6months": [0.1309, 0.1739],
    "less_vol_interpolated_3months": [0.2099, 0.1830],
    "vol_interpolated": [0.4734, 0.3282],
    "vol_interpolated_6months": [0.2888, 0.3094],
    "vol_interpolated_3months": [0.5489, 0.3154],}

# kappa, theta, xi, rho, v0, mu
Heston_paparams = {
    "less_vol_interpolated": 
        [1.5068, 0.1543, 0.2871, -0.0932, 0.0337, 0.1164],
    "less_vol_interpolated_6months": 
        [1.4450, 0.1547, 0.2634, -0.0826, 0.0317, 0.1309],
    "less_vol_interpolated_3months": 
        [1.6336, 0.1301, 0.2926, -0.0018, 0.0357, 0.2099],
    "vol_interpolated": 
        [1.6678, 0.1703, 0.2893, 0.1315, 0.1153, 0.3282],
    "vol_interpolated_6months": 
        [1.2320, 0.1557, 0.2652, 0.0415, 0.1022, 0.2888],
    "vol_interpolated_3months": 
        [1.5123, 0.1624, 0.2914, 0.0104, 0.1070, 0.5489]}


# mu, lamb, nu, omega, sigma
# sigma was not estimated with MLE since it didn't appear in the paper.
# we use the assumption that sigma = sqrt(var(log_returns) / dt)
Merton_paparams = {
    "less_vol_interpolated": 
        [0.1164, 0.7114, -0.0081, 0.0786, 0.2216388407804059],
    "less_vol_interpolated_6months": 
        [0.1309, 4.6730, 0.0054, 0.0735, 0.21622953579100895],
    "less_vol_interpolated_3months": 
        [0.2099, 5.3020, 0.0056, 0.0681, 0.23218320400155792],
    "vol_interpolated": 
        [0.3282, 0.4912, -0.0006, 0.0808, 0.4027095127433022],
    "vol_interpolated_6months": 
        [0.2388, 0.7176, -0.0009, 0.0804, 0.3791527039693616],
    "vol_interpolated_3months": 
        [0.5489, 0.6734, 0.0218, 0.0705, 0.3927754953312561]}


# kappa, theta, xi, rho, v0, mu, lambda, mu_J, sigma_J
SVJ_paparams = {
    "less_vol_interpolated": 
        [1.6549, 0.1557, 0.2806, 0.0257, 0.0336, 0.1055, 
            0.6263, -0.0062, 0.2483],
    "less_vol_interpolated_6months": 
        [1.5861, 0.1508, 0.2948, -0.0065, 0.0319, 0.1183, 
            0.5295, 0.0182, 0.2345],
    "less_vol_interpolated_3months": 
        [1.3898, 0.1715, 0.2428, -0.1481, 0.0360, 0.2015, 
            0.5171, -0.0058, 0.2405],
    "vol_interpolated": 
        [1.7951, 0.1702, 0.2746, 0.0321, 0.1153, 0.4695, 
            0.5338, -0.0160, 0.2565],
    "vol_interpolated_6months": 
        [1.3532, 0.1705, 0.2728, 0.0008, 0.1022, 0.2856, 
            0.4286, -0.0199, 0.3172],
    "vol_interpolated_3months": 
        [1.7624, 0.1823, 0.2928, -0.0278, 0.1070, 0.5479, 
            0.5194, 0.0295, 0.2136]}