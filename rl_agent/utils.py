import numpy as np
import torch

class myOptimizer():

    def __init__(self, lr, mu, mu_square, adaptation_rate, transaction_cost):
        self.lr = lr
        self.mu = mu
        self.mu_square = mu_square
        self.adaptation_rate = adaptation_rate
        self.transaction_cost = transaction_cost

    def step(self, grad_n, reward, last_observation, last_gradient):

        numerator = self.mu_square - (self.mu * reward)
        denominator = np.sqrt((self.mu_square - (self.mu ** 2)) ** 3)

        gradient = numerator / denominator

        current_grad = (-1.0 * self.transaction_cost * grad_n)

        previous_grad = (last_observation + self.transaction_cost) * last_gradient

        gradient = torch.as_tensor(gradient) * (current_grad + previous_grad)

        return torch.as_tensor(self.lr * gradient)

    def after_step(self, reward):

        self.mu = self.mu + self.adaptation_rate * (reward - self.mu)
        self.mu_square = self.mu_square + self.adaptation_rate * ((reward ** 2) - self.mu_square)
        



