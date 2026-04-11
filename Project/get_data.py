import yfinance
import pandas as pd
import numpy as np
import pickle
import os
import pandas as pd


def get_log_returns(data, index):
    if index == 0:
        d = data['Close']
    else:
        d = data[f'Close.{index}']

    n = len(d)
    X_list = []

    for i in range(3, n):
        X_i = np.log(float(d.iloc[i]) / float(d.iloc[i-1]))
        X_list.append(X_i)
    
    return X_list


if __name__=='__main__':

    less_vol = "VOD.L, MSFT, BP.L, LLOY.L, AAPL, IBM, CMCSA, FDX, PEP, MCD, HSBA.L, GS, WMT, LAND.L, GRI.L, SHEL.L, NG.L, TSCO.L, NXT.L, JNJ, PFE, BT-A.L".split(", ")
    
    # replace deliveroo ROO.L with doordash DASH
    # replace International Distribution Services Limited IDS.L with FedEx FDX
    vol = "TSLA, SHOP, SNOW, DASH, PLTR, META, UBER, FDX, ABNB, EXPE, BARC.L, ADBE, AMZN, ZM, FOXT.L, SVS.L, ORCL, GOOGL, MKS.L, JD.L, MRNA, BNTX".split(", ")


    new_data = "CAT, CSX, ROK, BSX, CI, DHR, AMAT, MU, QCOM, EIX, PPL, XEL, BK, KEY, TRV, NUE, PPG, VMC, CPB, KR, TGT, DVN, SLB, VLO, EA, T, VZ, MAR, TJX, YUM, EQR, PLD, WY".split(", ")


    start_date = "2023-10-01"
    end_date = "2024-01-01"

    # download the data
    data = yfinance.download(new_data, start=start_date, end=end_date)

    # number of na vals in the data
    missing_data = len(data[data.isna().any(axis=1)])

    # make sure there are no missing values
    if missing_data > 0:
        print("There are missing values in the dataset!", missing_data)
        
    df = pd.DataFrame(data)
    df.to_csv(r'new_data_3months.csv')


    data = pd.read_csv('new_data_3months.csv')

    num_stocks = 33
    log_returns = []
    for i in range(num_stocks):
        log_returns.extend(get_log_returns(data, i))

    res = pd.DataFrame({"Log_Returns": log_returns})

    res.to_csv("new_data_log_returns_3months.csv", index=False)

