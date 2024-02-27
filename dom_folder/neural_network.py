import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def error(y, y_tilde):
    return np.mean(np.square(y - y_tilde))

# Initiate data, weights and biases
x = np.linspace(1, 6, 50).reshape(-1, 1)
y = x**2

n_samples = x.shape[0]
n_features = x.shape[1]
n_hidden_neurons = 4
n_categories = 1

# Initialize the weights with small random numbers, scaled by the square root of the number of weights
hidden_w = np.random.randn(n_features, n_hidden_neurons)
output_w = np.random.randn(n_hidden_neurons, n_categories)
hidden_w = np.random.randn(n_features, n_hidden_neurons)/np.sqrt(hidden_w.size)
hidden_b = np.zeros((1,n_hidden_neurons)) + 0.01

output_w = np.random.randn(n_hidden_neurons, n_categories)/np.sqrt(output_w.size)
output_b = np.zeros((1,n_categories)) + 0.01

# Normalize the labels for sigmoid activation
y = (y - np.min(y)) / (np.max(y) - np.min(y))

#Feed forward
def feed_forward(X):
    z_h = np.matmul(X, hidden_w) + hidden_b
    a_h = sigmoid(z_h)

    z_o = np.matmul(a_h, output_w) + output_b
    a_o = sigmoid(z_o)

    return a_o, a_h, z_o, z_h

def backpropagation(X, Y):
    a_o, a_h, z_o, z_h = feed_forward(X)
    print(a_o)

    delta_L = (a_o - Y) * sigmoid_derivative(z_o)
    delta_h = np.matmul(delta_L,output_w.T) * sigmoid_derivative(z_h)

    return delta_L, delta_h

def training(eta, output_w, output_b, hidden_w, hidden_b, X):

    for i in range(10000):
        if i % 1000 == 0:
            y_tilde = feed_forward(x)[0]
            print(error(y, y_tilde))


        a_o, a_h, z_o, z_h = feed_forward(x)
        delta_L, delta_h = backpropagation(x,y)

        output_w -= eta * np.matmul(a_h.T, delta_L)
        output_b -= eta * np.sum(delta_L, axis = 0)

        hidden_w -= eta * np.matmul(x.T, delta_h)
        hidden_b -= eta * np.sum(delta_h, axis = 0)

training(0.1, output_w, output_b, hidden_w, hidden_b, x)

y_tilde = feed_forward(x)[0]
plt.plot(x, y)
plt.plot(x, y_tilde)

plt.show()
