#Importz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#Getting them dataz
URL = \
   'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'

df = pd.read_csv(URL)
#kutter av piss
df_ = df.drop(columns=['species']).copy()

#definerer gausisk funskjon
def gaussian_activation(value, mean, std_dev):
    return np.exp(-((value - mean) ** 2) / (2 * std_dev ** 2))


std_dev = 0.1
latencies = list()
for column in df_.columns: #går gjennom hver kolonne
    feature_values = df_[column] #liste med alle verdier fra kolonne
    means = np.linspace(feature_values.min(), feature_values.max(), 10) #liste med means

    feature_latencies_list = list() #orker ikke å kommentere lenger gl
    for value  in feature_values:
        neuron_latencies = []
        for mean in means:
            activation = gaussian_activation(value, mean, std_dev)
            if activation > 0.1:
                latency = 1 - activation
            else:
                latency = 'NaN'

            neuron_latencies.append(latency)
        feature_latencies_list.append(neuron_latencies)
    latencies.append(feature_latencies_list)
latencies = np.array(latencies)


print(np.array(latencies).shape)

reshaped_matrix = latencies.transpose(1, 0, 2).reshape(150, -1).T

print(reshaped_matrix.shape)

lat = pd.DataFrame(reshaped_matrix)
print(lat)
