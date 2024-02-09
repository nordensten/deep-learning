# import necessary packages
import numpy as np
import matplotlib.pyplot as plt


def feed_forward(x):
    # x is a nx1 vector

    # weighted sum of inputs to the output layer
    z_vec = np.matmul(w_mat,x) + b_vec
    # Output from output node (one node only)
    # Here the output is equal to the input
    return z_vec

def backpropagation(x, y):
    a_L = feed_forward(x)
    # derivative of cost function
    derivative_cost = a_L - y
    # the variable delta in the equations, note that output a_1 = z_1, its derivatives wrt z_o is thus 1
    delta_1 = derivative_cost
    # gradients for the output layer
    output_weights_gradient = delta_1*x
    output_bias_gradient = delta_1
    # The cost function is 0.5*(a_1-y)^2. This gives a measure of the error for each iteration
    return output_weights_gradient, output_bias_gradient

# ensure the same random numbers appear every time
np.random.seed(0)
# Defining the neural network

N = 1000

# Input variable
x = np.expand_dims(np.linspace(0,1,N),axis=1)

print(x.shape)

# Target values
y = 4*x**2 + 2*x + 1.0 + 99*x**4

n_inputs = x.shape
n_outputs = 1

# Initialize the network
# weights and bias in the output layer
w_mat = np.random.randn(N,N)
b_vec = np.random.randn(N,1)
print(b_vec.shape)

# implementing a simple gradient descent approach with fixed learning rate
eta = 0.001
for i in range(1000):
    # calculate gradients from back propagation
    derivative_w1, derivative_b1 = backpropagation(x, y)
    # update weights and biases
    w_mat -= eta * derivative_w1
    b_vec -= eta * derivative_b1

# our final prediction after training
ytilde = np.matmul(w_mat,x) + b_vec
print(0.5*((ytilde-y)**2))

#plt.plot(x,y)
plt.plot(x,ytilde)
plt.plot(x,y)
plt.show()
