import numpy as np
import pandas as pd
import torch

class Environment:
    
    def __init__(self, data, history_t=8, history=[0.1, 0.2, -0.1, -0.2, 0., 0.5, 0.9], state_size=9):
        self.data = data
        self.history = history
        self.history_t = history_t
        self.state_size = state_size
        self.cost_rate = 0.0001
        self.reset()
        
    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.position_value = 0.
        self.history = self.history[:self.state_size - 1]
        return [self.position_value] + self.history # obs
    
    def step(self, act):
        reward = 0
        
        # act = 0: stay, act > 0: buy, act < 0: sell
        #Additive profits
        cost_amount = np.abs(act-self.position_value)
        
        Zt = self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t-1), :]['Close']
        reward = (self.position_value * Zt) - (self.cost_rate * cost_amount)
        self.profit = self.position_value * Zt
        self.profits += self.profit

        # set next time
        self.t += 1
        self.position_value = act
        
        self.history.pop(0)
        
        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t-1), :]['Close']) # the price being traded
        
        self.position_value = self.position_value.item()

        return [self.position_value] + self.history, reward, self.done # obs, reward, done




if __name__ == "__main__":
    data = pd.read_csv('./data/EURUSD_Candlestick_1_M_BID_01.01.2021-04.02.2023.csv')
    # data['Local time'] = pd.to_datetime(data['Local time'])
    data = data.set_index('Local time')
    print(data.index.min(), data.index.max())

    date_split = '19.09.2022 17:55:00.000 GMT-0500'
    train = data[:date_split]
    test = data[date_split:]
    print(train.head(10))

    history = []
    for i in range(1, 9):
        c = train.iloc[i, :]['Close'] - train.iloc[i-1, :]['Close']
        history.append(c)

    env = Environment(train, history=history)
    print(env.reset())
    for _ in range(9, 12):
        pact = np.random.randint(3)
        print(env.step(pact)[1])
    
