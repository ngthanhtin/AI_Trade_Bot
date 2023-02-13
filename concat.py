import pandas as pd

data1 = pd.read_csv('./data/EURUSD_Candlestick_1_M_BID_01.01.2018-31.12.2020.csv')
data2 = pd.read_csv('./data/EURUSD_Candlestick_1_M_BID_01.01.2021-04.02.2023.csv')    
data1 = data1.set_index('Local time')
data2 = data2.set_index('Local time')

print(len(data1), len(data2))
result = pd.concat([data1, data2])
print(len(result))
result.to_csv('abc.csv',index=True)
