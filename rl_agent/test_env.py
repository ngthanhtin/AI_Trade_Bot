from env import Environment
from policy import Policy
from utils import myOptimizer

import pandas as pd
import numpy as np
import torch
from collections import OrderedDict

import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter('runs/new_data_ex_7')

    data = pd.read_csv('./data/EURUSD_Candlestick_1_M_BID_01.01.2021-04.02.2023.csv')
    # data = pd.read_csv('./data/EURUSD_Candlestick_30_M_BID_01.01.2021-04.02.2023.csv')
    # data['Local time'] = pd.to_datetime(data['Local time'])
    data = data.set_index('Local time')
    print(data.index.min(), data.index.max())

    date_split = '19.09.2022 17:55:00.000 GMT-0500'
    # date_split = '25.08.2022 04:30:00.000 GMT-0500' # 30 min
    # date_split = '03.02.2023 15:30:00.000 GMT-0600' # 30 min

    train = data[:date_split]
    test = data[date_split:]


    learning_rate = 0.001
    first_momentum = 0.0
    second_momentum = 0.0001
    transaction_cost = 0.0001
    adaptation_rate = 0.01
    state_size = 15
    equity = 1.0

    agent = Policy(input_channels=state_size)
    optimizer = myOptimizer(learning_rate, first_momentum, second_momentum, adaptation_rate, transaction_cost)

    

    history = []
    for i in range(1, state_size):
        c = train.iloc[i, :]['Close'] - train.iloc[i-1, :]['Close']
        history.append(c)

    env = Environment(train, history=history, state_size=state_size)
    observation = env.reset()
    

    model_gradients_history = dict()
    checkpoint = OrderedDict()

    for name, param in agent.named_parameters():
        model_gradients_history.update({name: torch.zeros_like(param)})



    for i in tqdm(range(state_size, len(train))):
        observation = torch.as_tensor(observation).float()
        action = agent(observation)
        observation, reward, _ = env.step(action.data.to("cpu").numpy())


        
        
        action.backward()

        for name, param in agent.named_parameters():
        
            grad_n = param.grad
            param = param + optimizer.step(grad_n, reward, observation[-1], model_gradients_history[name])
            checkpoint[name] = param
            model_gradients_history.update({name: grad_n})

        if i > 10000:
            equity += env.profit
            writer.add_scalar('equity', equity, i)    
        else:
            writer.add_scalar('equity', 1.0, i) 

        optimizer.after_step(reward)
        agent.load_state_dict(checkpoint)

    ###########
    ###########

    # history = []
    # for i in range(1, state_size):
    #     c = test.iloc[i, :]['Close'] - test.iloc[i-1, :]['Close']
    #     history.append(c)

    # env = Environment(test, history=history, state_size=state_size)
    # observation = env.reset()
    

    # model_gradients_history = dict()
    # checkpoint = OrderedDict()

    # for name, param in agent.named_parameters():
    #     model_gradients_history.update({name: torch.zeros_like(param)})

    # for _ in tqdm(range(state_size, len(test))):
    #     observation = torch.as_tensor(observation).float()
    #     action = agent(observation)
    #     observation, reward, _ = env.step(action.data.numpy())


        
        
    #     action.backward()

    #     for name, param in agent.named_parameters():
        
    #         grad_n = param.grad
    #         param = param + optimizer.step(grad_n, reward, observation[-1], model_gradients_history[name])
    #         checkpoint[name] = param
    #         model_gradients_history.update({name: grad_n})

    #     optimizer.after_step(reward)
    #     agent.load_state_dict(checkpoint)

    print(env.profits)