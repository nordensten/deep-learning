import numpy as np 
import matplotlib.pyplot as plt 

def activation_function(z):
    return 1/(1+np.exp(-z))

def cost_function(y,y_tilde):
    return (1/len(y))*np.sum((y - y_tilde)**2)

def node_activation(input, w, b):
    return w*input + b

w = np.array([1.2, 3.7])
b = np.array([0, 0])


x = np.linspace(0,1,10)
y = np.zeros(10)
y[5] = 1

y_tilde = np.zeros(10)

for i in range(len(x)):
    z1 = node_activation(x[i],w[0],b[0])
    act_func = activation_function(z1)
    z2 = node_activation(act_func,w[1],b[1])



    




plt.plot(x,y,'o')
plt.plot(x,y_tilde)
plt.show()





