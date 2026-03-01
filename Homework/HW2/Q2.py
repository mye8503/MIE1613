import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

import SimFunctions
import SimRNG
import SimClasses


from pprint import pprint 



# intermediate function, splits data into day+time bins and counts number of arrivals per bin
def get_arrival_counts(K, data):
    # array to hold all days, with bins for each bin length (hour or half-hour, haven't finalized yet)
    # 32 rows to account for zero-indexing
    days = np.zeros((32, K))

    # cycle through all days in december
    for day in range(1, 32):
        day_data = data[pd.to_datetime(data['Start Time']).dt.day == day]

        for time_bin in range(K):
            bin_data = day_data[pd.to_datetime(day_data['Start Time']).dt.hour == time_bin]

            days[day][time_bin] = len(bin_data)

    return days


# ========== PART A =========== #
def get_lambda(K, days):
    # one lambda calculation per time bin
    _lambda = np.zeros((K,))
    for time_bin in range(K):
        # cut off first row since day '0' doesn't exist
        _lambda[time_bin] = np.mean(days[1:, time_bin])

    return(_lambda)


# ========== PART B ========== #
# from lecture: we can use a Poisson process model if the mean is equal 
# to the variance: E(N(t)) ~= Var(N(t)) --> Var(N(t)) / E(N(t)) ~= 1
# 
# step 1: for each day j, calculate the cumulative number of arrivals 
# by time bin t, C_j(t)
# 
# step 2: average over k=31 days to get Lambda(t)
# 
# step 3: calculate the variance with 
# V(t) = 1/(k-1) (sum_over_k( C_j(t) - Lambda(t))^2 )
# 
# step 4: get the ratio of V(t) / Lambda(t), then average over
# t = [0...K] to get the ratio over all days and hours
def variance_to_mean(K, days):
    # cumulative sum over columns
    C = np.cumsum(days, axis=1)


    _Lambda = np.zeros((K,))
    V = np.zeros((K,))
    v_over_m = 0
    for time_bin in range(K):
         # Lambda(t) calculation
        _Lambda[time_bin] = np.mean(C[1:, time_bin])

        # Variance calculation
        for day in range(1, 32):
            V[time_bin] += np.power(C[day][time_bin] - _Lambda[time_bin], 2)
        V[time_bin] = V[time_bin] / 30

        # variance-to-mean ratio calculation
        v_over_m += V[time_bin] / _Lambda[time_bin]
    
    v_over_m = v_over_m / K
    return v_over_m


# ========== PART C ========== #
# from lecture:
#
# step 0: choose lambda_max = max_t(lambda(t)) 
# 
# step 1: set indices n=1, k=1, S_0=0
# 
# step 2: generate A_n with exponential distribution and mean
# lambda_max. this inter-arrival time will be less than or 
# equal to the actual inter-arrival time since we used the 
# maximum rate. to fix this, we [COMPLETE...]
#
# step 3: calculate S_n = S_(n-1) + A_n
#
# step 4: generate U with uniform distribution between (0,1)
#
# step 5: if U <= lambda(S_n) / lambda_max, accept
# S_k = S_n
# A_k = S_k - S_(k-1)
# k = k + 1
#
# step 6: n = n + 1
# 
# step 7: repeat step 2


###############################################################
######### PYTHONSIM INITIALIZATION AND PARAMETERS #############
###############################################################

# Initialization
SimClasses.Clock = 0
SimRNG.ZRNG = SimRNG.InitializeRNSeed()
Calendar = SimClasses.EventCalendar()
NumUnhappyCustomers = SimClasses.DTStat()
Station = SimClasses.Resource()
# station --> 'idle' resources means empty bike holders (no bikes)
#         --> 'busy' resources means bikes are being held
RunLength = 1440.0 # run length in minutes (1 day)

# Example-specific statistics
UnhappyCustomersPerDay = []

# Parameters
# N: number of experiment replications (days)
N = 200
# B: number of bikes at the station (init 25)
B_init = 25
Capacity = 40


###############################################################
################## SIMULATION FUNCTIONS #######################
###############################################################
def RentalRequest():
    '''
    Called when a customer wants to rent a bike
    checks if B >= 1, if yes, then B = B - 1
    if B == 0, increase unhappy number count by 1
    '''
    if Station.CurrentNumBusy >= 1:
        Station.Free(1) # remove bike from station
    
    else:
        # we don't really need to keep track of the time a 
        # customer was made unhappy or anything, just the total
        # number
        #
        # so pass in 0 as the parameter
        NumUnhappyCustomers.Record(0)


def ReturnRequest():
    '''
    Called when a customer wants to return a bike
    checks if B <= C - 1, if yes, then B = B + 1
    if B == C, increase unhappy number count by 1
    '''   
    if Station.CurrentNumBusy <= Station.NumberOfUnits - 1:
        Station.Seize(1) # add bike to station
    
    else:
        # we don't really need to keep track of the time a 
        # customer was made unhappy or anything, just the total
        # number
        #
        # so pass in 0 as the parameter
        NumUnhappyCustomers.Record(0) 


if __name__ == "__main__":
    # hour bins
    K = 24

    # Load the data from the uploaded file
    file_path_start = 'BikeShare_Trips_Start.csv'
    file_path_end = 'BikeShare_Trips_End.csv'

    # ========== PART A ========== #
    data = pd.read_csv(file_path_start)
    start_days = get_arrival_counts(K, data)

    data = pd.read_csv(file_path_end)
    end_days = get_arrival_counts(K, data)

    lambda_d = get_lambda(K, start_days)
    lambda_r = get_lambda(K, end_days)

    print(f"lambda_d: {lambda_d}\nlambda_r: {lambda_r}")

    # ========== PART B ========== #
    vtm_start = variance_to_mean(K, start_days)
    vtm_end = variance_to_mean(K, end_days)

    print(f"rental vtm: {vtm_start}\nreturn vtm: {vtm_end}")

    # ========== PART D ========== #
    for reps in range(N):
        # Initialize simulation objects and
        #   schedule all rentals, returns, end-of-simulation event,
        #   and event to clear statistics after warm-up period
        SimFunctions.SimFunctionsInit(Calendar)

        # schedule all rental times (in minutes) using thinning method
        lambda_d_max = np.max(lambda_d)
        S = 0
        S_next = 0
        while S < RunLength:
            A = SimRNG.Expon(lambda_d_max, 2)
            S_next = S + A

            # lambda_d_max is the max rate of event occurrences --> the
            # min time between event occurences. if S_next > RunLength,
            # the probability that S > RunLength (S generated with the 
            # correct lambda) is pretty much guaranteed.
            if S_next > RunLength: break

            U = SimRNG.Uniform(0, 1, 1)

            # get the lambda of the time bin that S_next belongs to
            # time bins are in hours, S_next is in minutes, so need to
            # convert
            if U <= lambda_d[int(np.floor(S_next/60))] / lambda_d_max:
                S = S_next
                SimFunctions.Schedule(Calendar, "RentalRequest", A)

        # schedule all return times (in minutes) using thinning method
        lambda_r_max = np.max(lambda_r)
        S = 0
        S_next = 0
        while S < RunLength:
            A = SimRNG.Expon(lambda_r_max, 2)
            S_next = S + A

            if S_next > RunLength: break

            U = SimRNG.Uniform(0, 1, 1)

            # get the lambda of the time bin that S_next belongs to
            if U <= lambda_r[int(np.floor(S_next/60))] / lambda_r_max:
                S = S_next
                SimFunctions.Schedule(Calendar, "ReturnRequest", A)

        # end of day
        SimFunctions.Schedule(Calendar,"EndSimulation",RunLength)

        # init station
        Station.SetUnits(Capacity)
        Station.Seize(B_init)

        # Main simulation loop
        while Calendar.N() > 0:
            NextEvent = Calendar.Remove()
            SimClasses.Clock = NextEvent.EventTime
            if NextEvent.EventType == "RentalRequest":
                RentalRequest()
            elif NextEvent.EventType == "ReturnRequest":
                ReturnRequest()
            elif NextEvent.EventType == "EndSimulation":
                break
        
        UnhappyCustomersPerDay.append(NumUnhappyCustomers.N())
    
    # print(UnhappyCustomersPerDay)

    output = pd.DataFrame(
        {"UnhappyCustomersPerDay": UnhappyCustomersPerDay}
    )

    print("Mean") 
    print(output.mean())
        
    print("95% CI Half-Width")
    print(1.96*np.sqrt(output.var(ddof=1)/len(output)))


    # N = 200: 8.755 +/- 1.588154
    # N = 1000: 8.856 +/- 0.692732