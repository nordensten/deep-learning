import numpy as np
import matplotlib.pyplot as plt
import math
import autograd.numpy as np
import sys
import warnings
from autograd import grad, elementwise_grad
from random import random, seed
from copy import deepcopy, copy
from typing import Tuple, Callable
from sklearn.utils import resample

class FFNN:
    def __init__(self, input, target):#, dimensions, hidden_func, output_func, cost_func):
        # self.dimensions = dimensions
        # self.hidden_func = hidden_func
        # self.output_func = output_func
        # self.cost_func = cost_func

        input = input.reshape(-1, 1)
        target = target.reshape(-1, 1)
        normalized_x = (input - np.min(input)) / (np.max(input) - np.min(input))
        normalized_y = (target - np.min(target)) / (np.max(target) - np.min(target))
        self.X = normalized_x
        self.Y = normalized_y

        self.weights = list()
        self.bias = list()
        self.a_matrices = list()
        self.z_matrices = list()

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def error(self, y, y_tilde):
        return np.mean(np.square(y - y_tilde))

    def initialize(self):
        self.n_samples = self.X.shape[0]
        self.n_input = self.X.shape[1]
        self.n_hidden_neurons = 4
        self.n_categories = 1

        # print(f"input shape:", self.X.shape)
        # print(f"Target shape:", self.Y.shape)

        self.hidden_w = np.random.randn(self.n_input, self.n_hidden_neurons)
        self.output_w = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.hidden_b = np.zeros((1,self.n_hidden_neurons)) + 0.01
        self.output_b = np.zeros((1,self.n_categories)) + 0.01

        # print(f"hidden layer shape:", self.hidden_w.shape)
        # print(f"output layer shape:", self.output_w.shape)
        # print(f"hidden bias:", self.hidden_b.shape)
        # print(f"output bias:", self.output_b.shape)

        #print(self.hidden_w.shape)


    def feedforward(self):
        #print(self.hidden_w)
        z_h = np.dot(self.X, self.hidden_w) + self.hidden_b
        # print(f"z_h:", z_h.shape)

        a_h = self.sigmoid(z_h)
        # print(f"a_h:", a_h.shape)

        z_o = np.dot(a_h, self.output_w) + self.output_b

        # print(f"z_o:", z_o.shape)
        a_o = self.sigmoid(z_o)
        # print(f"s_o:", a_o.shape)

        return a_o, z_o, z_h, a_h


    def backpropagation(self):
        eta = 0.1
        for i in range(10000):
            a_o, z_o, z_h, a_h = self.feedforward()

            delta_L = (a_o - self.Y) * self.sigmoid_derivative(z_o)
            # print(f"delta_L:", delta_L.shape)
            delta_h = np.dot(delta_L, self.output_w.T) * self.sigmoid_derivative(z_h)
            # print(f"delta_h:", delta_h.shape)

            gradient_w_o = np.dot(a_h.T, delta_L)
            # print(f"gradient_w_o:", gradient_w_o.shape)

            gradient_w_h = np.dot(self.X.T, delta_h)
            # print(f"X.T:", self.X.T.shape)
            # print(f"gradient_w_h:", gradient_w_h.shape)

            gradient_b_o = np.sum(delta_L, axis=0)
            gradient_b_h = np.sum(delta_h, axis=0)
            # print(f"gradient_b_h:", gradient_b_h.shape)
            # print(f"gradient_b_o:", gradient_b_o.shape)



            self.output_w -= eta * gradient_w_o
            self.hidden_w -= eta * gradient_w_h
            self.output_b -= eta * gradient_b_o
            self.hidden_b -= eta * gradient_b_h
            #print(gradient_w_o)

            y_tilde = self.feedforward()[0]
            print(self.error(self.Y, y_tilde))
        plt.plot(self.X, self.Y)
        plt.plot(self.X, y_tilde)
        plt.show()









if __name__ == '__main__':
    x = np.linspace(1,10,500)
    y = np.sin(x)
    start = FFNN(x, y)
    start.initialize()
    start.feedforward()
    start.backpropagation()
