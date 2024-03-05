#Importsssss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
#import warnings
#warnings.filterwarnings("ignore")

#importing the dataset:

URL = \
   'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'

df = pd.read_csv(URL)
#print(df.head())
print(df)
df_ = df.drop(columns=['species']).copy()

df_.head()

#print(df_.shape)

# df_.plot.hist(alpha = 0.4, figsize = (12, 4))
# plt.legend(title = "Dataset cilumns:" ,bbox_to_anchor = (1.0, 0.6),
#                                                    loc = 'upper left')
# plt.title('Iris dataset', fontsize = 20)
# plt.xlabel('Input value', fontsize = 15)
# plt.show()



def Gaus_neuron(df, n, step, s):
    neuron_list = list()
    x_axis_list = list()
    t = 0

    for col in df.columns:

        vol = df[col].values
        min_ = np.min(vol)
        max_ = np.max(vol)
        x_axis[0] = min_
        x_axis[-1] = max_
        x_axis_list.append(np.round(x_axis, 10))
        neurons = np.zeros((n, len(x_axis)))

        for i in range(n):

            loc = (max_ - min_) * (i /(n-1)) + min_
            neurons[i] = norm.pdf(x_axis, loc, s[t])
            neurons[i] = neurons[i] / np.max(neurons[i])

        neurons_list.append(neurons)
        t += 1

    return neurons_list, x_axis_list


sigm = [0.1, 0.1, 0.2, 0.1]
d = Gaus_neuron(df_, 10, 0.001, sigm)


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

fig.set_figheight(8)
fig.set_figwidth(10)

k = 0

for ax in [ax1, ax2, ax3, ax4]:

    ax.set(ylabel = f'{df_.columns[k]} \n\n Excitation of Neuron')

    for i in range(len(d[0][k])):

        ax.plot(d[1][k], d[0][k][i], label = i + 1)

    k+=1

plt.legend(title = "Presynaptic neuron number \n      in each input column" ,
                            bbox_to_anchor = (1.05, 3.25), loc = 'upper left')
plt.suptitle(' \n\n  Gaussian receptive fields for Iris dataset',
                                                                 fontsize = 15)
ax.set_xlabel(' Presynaptic neurons and\n input range of value feature',
                                                  fontsize = 12, labelpad = 15)

plt.show()
