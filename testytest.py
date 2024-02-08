import numpy as np
import matplotlib.pyplot as plt

#define weights
w01 = 0.5
w02 = 0.5
w11 = 0.5
w12 = 0.5

#first iteration
x1 = 1
y1 = 1

z01 = x1*w01
z11 = x1*w11

#activation
a01 = 1/(1 + np.exp(-z01))
a11 = 1/(1 + np.exp(-z11))

z02 = a01*w11
z12 = a11*w12


y_tilda = z02 + z12

print(y_tilda)
