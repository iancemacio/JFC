import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

try:
    # Use the provided GitHub link to read the CSV
    JFCData = pd.read_csv('https://raw.githubusercontent.com/iancemacio/JFC/main/JFC.csv',header=0, usecols=['Date', 'Close'],parse_dates=True,index_col='Date')
    print(JFCData.info())
    print(JFCData.head())
    print(JFCData.tail())
    print(JFCData.describe())

    plt.figure(figsize=(10,5))
    plt.plot(JFCData)
    plt.show()

    JFCDataPctChange = JFCData.pct_change()

    JFCLogReturns = np.log(1 + JFCDataPctChange)
    print(JFCLogReturns.tail(10))

    plt.figure(figsize=(10,5))
    plt.plot(JFCLogReturns)
    plt.show()

    MeanLogReturns = np.array(JFCLogReturns.mean())

    VarLogReturns = np.array(JFCLogReturns.var())

    StdevLogReturns = np.array(JFCLogReturns.std())


    Drift = MeanLogReturns - (0.5 * VarLogReturns)
    print("Drift = ",Drift)

    NumIntervals = 2515

    Iterations = 20

    np.random.seed(7)
    SBMotion = norm.ppf(np.random.rand(NumIntervals, Iterations))



    DailyReturns = np.exp(Drift + StdevLogReturns * SBMotion)


    StartStockPrices = JFCData.iloc[0]

    StockPrice = np.zeros_like(DailyReturns)

    StockPrice[0] = StartStockPrices

    for t in range(1, NumIntervals):

        StockPrice[t] = StockPrice[t - 1] * DailyReturns[t]



    plt.figure(figsize=(10,5))

    plt.plot(StockPrice)

    JFCTrend = np.array(JFCData.iloc[:, 0:1])

    plt.plot(JFCTrend,'k*')

    plt.show()

except FileNotFoundError:
    print("Error: 'JFC.csv' not found. Please make sure the file is in the correct directory.")
except Exception as e:
    print(f"An error occurred: {e}")
