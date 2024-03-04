import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# def func(x):
#     return 1/(1 + np.exp(x))
#
# df_db = jax.grad(func)
#
# x = 1.0
# a = 2.0
# b = 3.0
# derivative = df_db(x)
# print(derivative)


import jax.numpy as jnp
import jax

def sigmoid(x, scale=100):
    return 1 / (1 + jnp.exp(-x*scale))

print(jax.grad(sigmoid)(-1000.0))
# nan
#
# def error(a, y):
#     return 0.5*(a - y)**2
#
# def sigmoid(z):
#     return 1/(1 + np.exp(z))
#
# def weighting(x, weight, bias):
#     return x*weight + bias
#
# def feed_forward(x):
#     z = weighting(x, weight, bias)
#     a = sigmoid(z) #We drop the sigmoid function
#     return a, z
#
# def backpropagation(x, y):
#     a, z = feed_forward(x)
#
#
#     #Now the derivatives
#     weighting_der_b = jax.grad(weighting, argnums=2)
#     weighting_der_a = jax.grad(weighting, argnums=1)
#     sigmoid_der = jax.grad(sigmoid)
#     error_der = jax.grad(error, argnums=0)
#
#     dCda = error_der(a, y)
#     dadz = sigmoid_der(z)
#     dzdw = weighting_der_a(weights)
#     dzdb = weighting_der_b(bias)
#
#     dCdw = dCda*dadz*dzdw
#     dCdb = dCda*dadz*dzdb
#
#     return dCdw, dCdb
#
#
#
#
# x = 4.0
# y = 2*x + 1.0
#
# weight = np.random.randn()
# bias = np.random.randn()
#
# eta = 0.1
# for i in range(30):
#     dCdw, dCdb = backpropagation(x, y)
#     weight -= eta*dCdw
#     bias -= eta*dCdb
#
#     y_tilde = weight*x + bias
#     print(error(y_tilde, y))
# # grad_square = jax.grad(square_function)
# #
# # x = 3.0
# # dvalue = grad_square(x)
# #
# # print(2*x)
# # print(dvalue)
