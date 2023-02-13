from env import Environment
from policy import Policy
from utils import myOptimizer

import pandas as pd
import numpy as np
import torch
from collections import OrderedDict

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    writer = SummaryWriter('runs/new_data_ex_3')


    # data = pd.read_csv('./data/EURUSD_Candlestick_1_M_BID_01.01.2021-04.02.2023.csv') # 1 min
    data = pd.read_csv('./data/EURUSD_Candlestick_1_M_BID_01.01.2021-04.02.2023.csv') # 30 min
    data = data.set_index('Local time')
    print(data.index.min(), data.index.max())

    # date_split = '19.09.2022 17:55:00.000 GMT-0500' # 1 min
    date_split = '25.08.2022 04:30:00.000 GMT-0500' # 30 min
    train = data[:date_split]
    test = data[date_split:]

    initial_money = 10.0
    
    learning_rate = 0.01
    first_momentum = 0.0
    second_momentum = 0.0
    transaction_cost = 0.0001
    adaptation_rate = 0.001
    state_size = 15

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
        observation, reward, _ = env.step(action.data.numpy())

        action.backward()

        for name, param in agent.named_parameters():
        
            grad_n = param.grad
            param = param + optimizer.step(grad_n, reward, observation[-1], model_gradients_history[name])
            checkpoint[name] = param
            model_gradients_history.update({name: grad_n})
        # print(optimizer.mu)
        optimizer.after_step(reward)
        agent.load_state_dict(checkpoint)
        writer.add_scalar('profits', env.profits, i)

    print(env.profits)