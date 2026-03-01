import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

import SimFunctions
import SimRNG
import SimClasses


from pprint import pprint 

# ========== PART A ========== #
# For each of the following outputs, determine whether they are a discrete-time output or
# a continuous-time output of the simulation: 
# 
# i. inventory level, 
# the inventory level changes based on order arrival times and demand times, which are
# both exponentially distributed variables. so the inventory level is a continuous-time
# output. 
# 
# ii. lost demand, 
# demands are satisfied immediately on arrival, which is an exponentially distributed
# variable. any lost demand is also calculated immediately as a function of the demand
# and the inventory level, so it is a continuous-time output.
#
# iii. ordering cost.
# the company reviews its inventory level and determines how many items to order at the
# beginning of each month. orders are only placed at the beginning of the month (assumed?),
# so ordering cost is a discrete-time output.



###############################################################
######### PYTHONSIM INITIALIZATION AND PARAMETERS #############
###############################################################

# Initialization
SimClasses.Clock = 0
SimRNG.ZRNG = SimRNG.InitializeRNSeed()
Calendar = SimClasses.EventCalendar()
InventoryLevel = SimClasses.CTStat()
LostDemand = SimClasses.CTStat()
OrderingCost = SimClasses.DTStat()
InventoryPosition = SimClasses.CTStat()
RunLength = 120 # run length in months

# Example-specific statistics
OrderingCostAvg = []
HoldingCostAvg = []
LostSalesCostAvg = []
TotalCostAvg = []

# Parameters
# N: number of experiment replications
N = 1000
# IPCondition: s given in question
IPCondition = [20, 20, 20, 30, 30, 30]
# MaxIP: S given in question
MaxIP = [50, 60, 70, 60, 80, 100]
# K: setup cost given in question
K = 32
# i: incremental cost given in question
i = 3
# h: holding cost per item per month
h = 1
# l: lost demand penalty
l = 12


###############################################################
################## SIMULATION FUNCTIONS #######################
###############################################################
def PlaceOrder(IPC, MIP):
    '''
    Called at the beginning of the month - company reviews its
    inventory level and places orders
    '''

    if InventoryPosition.Xlast < IPC:
        Z = MIP - InventoryPosition.Xlast
        # If the company orders Z items, it incurs a cost of
        # K +iZ, where K = $32 is the setup cost and i = $3 is 
        # the incremental cost per item ordered.
        OrderingCost.Record(K + i * Z)

        # Z = MIP - InventoryPosition.Xlast
        # InventoryPosition += Z --> InventoryPosition = MIP
        InventoryPosition.Record(MIP)

        OrderAmount = SimClasses.Entity()
        OrderAmount.ClassNum = Z

        # When an order is placed, the time required for it to arrive (or
        # lead time) is a random variable that is exponentially distributed 
        # with mean 0.75 month.
        SimFunctions.SchedulePlus(Calendar, "OrderArrived", SimRNG.Expon(0.75, 2), OrderAmount)

    # else: don't order anything, nothing needs to be done


def OrderArrived(OrderAmount):
    '''
    Called when an order arrives - update inventory level
    '''
    InventoryLevel.Record(InventoryLevel.Xlast + OrderAmount.ClassNum)



def GetDemandAmount():
    '''
    Helper function to determine the amount requested in the
    next demand
    '''

    # demand size = 1 w.p. 1/6, 2 w.p. 1/3, 3 w.p. 1/3, 4 w.p. 1/6
    # generate a uniform random variable between [0, 6] 
    # convert it to an integer with ceiling --> one of {1, 2, 3, 4, 5, 6}
    # if var == 1: demand = 1
    #        == 2 or == 3: demand = 2
    #        == 4 or == 5: demand = 3
    #        == 6: demand = 4
    U = np.ceil(SimRNG.Uniform(0, 6, 1))
    if U == 1: NextDemandAmount = 1
    elif U == 2 or U == 3: NextDemandAmount = 2
    elif U == 4 or U == 5: NextDemandAmount = 3
    else: NextDemandAmount = 4

    return NextDemandAmount


def Demand(DemandAmount):
    '''
    Called when a demand occurs - check inventory level and either
    fulfill the entire demand, or empty inventory and update lost
    demand. inventory position will also be updated
    '''
    if InventoryLevel.Xlast >= DemandAmount.ClassNum:
        InventoryLevel.Record(InventoryLevel.Xlast - DemandAmount.ClassNum)
        InventoryPosition.Record(InventoryPosition.Xlast - DemandAmount.ClassNum)

    else:
        LostDemand.Record(LostDemand.Xlast + DemandAmount.ClassNum - InventoryLevel.Xlast)
        InventoryPosition.Record(InventoryPosition.Xlast - InventoryLevel.Xlast)
        InventoryLevel.Record(0)

    # schedule next demand
    NextDemand = SimClasses.Entity()
    NextDemand.ClassNum = GetDemandAmount()

    # The times between demands are IID exponential random variables 
    # with a mean of 0.1 month.
    SimFunctions.SchedulePlus(Calendar, "Demand", SimRNG.Expon(0.1, 2), NextDemand)



if __name__ == "__main__":
    # ========== PART B ========== #
    # use zip to iterate over both lists at once
    for IPC, MIP in zip(IPCondition, MaxIP):
        # use common random numbers - same random numbers for each scenario
        # reset SimRNG numbers
        SimRNG.ZRNG = SimRNG.InitializeRNSeed()
        for reps in range(N):
            # Initialize simulation objects and
            #   schedule all rentals, returns, end-of-simulation event,
            #   and event to clear statistics after warm-up period
            SimFunctions.SimFunctionsInit(Calendar)

            # init inventory capacity and starting amount (50)
            InventoryLevel.Record(50)

            # schedule all order placements (beginning of month)
            for j in range(1, RunLength):
                SimFunctions.Schedule(Calendar, "PlaceOrder", j)

            # schedule first demand
            FirstDemand = SimClasses.Entity()
            FirstDemand.ClassNum = GetDemandAmount()

            # The times between demands are IID exponential random variables 
            # with a mean of 0.1 month.
            SimFunctions.SchedulePlus(Calendar, "Demand", SimRNG.Expon(0.1, 2), FirstDemand)

            # end of sim period
            SimFunctions.Schedule(Calendar,"EndSimulation", RunLength)


            # Main simulation loop
            while Calendar.N() > 0:
                NextEvent = Calendar.Remove()
                SimClasses.Clock = NextEvent.EventTime
                if NextEvent.EventType == "PlaceOrder":
                    PlaceOrder(IPC, MIP)
                elif NextEvent.EventType == "OrderArrived":
                    OrderArrived(NextEvent.WhichObject)
                elif NextEvent.EventType == "Demand":
                    Demand(NextEvent.WhichObject)
                elif NextEvent.EventType == "EndSimulation":
                    break

            HoldingCostAvg.append(h * InventoryLevel.Mean())
            LostSalesCostAvg.append(l * LostDemand.Mean())
            OrderingCostAvg.append(OrderingCost.Mean())
            TotalCostAvg.append(HoldingCostAvg[-1] + LostSalesCostAvg[-1] + OrderingCostAvg[-1])


        output = pd.DataFrame(
            {"HoldingCostAvg": HoldingCostAvg,
            "LostSalesCostAvg": LostSalesCostAvg,
            "OrderingCostAvg": OrderingCostAvg,
            "TotalCostAvg": TotalCostAvg}
        )

        print(f"For max inventory position {MIP} and inventory position condition {IPC}:")
        print(f"Mean {output.mean()["TotalCostAvg"]} +/- {1.96*np.sqrt(output.var(ddof=1)/len(output))["TotalCostAvg"]}\n")



# For max inventory position 50 and inventory position condition 20:
# Mean 531.5646623153817 +/- 15.747630311460215

# For max inventory position 60 and inventory position condition 20:
# Mean 560.0844386740357 +/- 11.911872660311463

# For max inventory position 70 and inventory position condition 20:
# Mean 586.5710934072893 +/- 10.246914665663773

# For max inventory position 60 and inventory position condition 30:
# Mean 544.532273919386 +/- 8.640126517095776

# For max inventory position 80 and inventory position condition 30:
# Mean 533.9535509727151 +/- 7.434019228954737

# For max inventory position 100 and inventory position condition 30:
# Mean 545.7227827934063 +/- 6.797999055319378