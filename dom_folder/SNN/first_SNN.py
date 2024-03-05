import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



class SNN_model:
    def __init__(self, data_URL, n_presynaptic_neurons):

        df = pd.read_csv(data_URL)
        self.n_presynaptic_neurons = n_presynaptic_neurons
        self.df = df.drop(columns=['species']).copy()
        self.step = 0.001


    def Gaussian_neurons(self):

        sigm = [0.1, 0.1, 0.2, 0.1]
        neurons_list = list()
        x_axis_list = list()
        t = 0

        for col in self.df.columns:

            vol = self.df[col].values
            min_ = np.min(vol)
            max_ = np.max(vol)
            x_axis = np.arange(min_, max_, self.step)
            x_axis[0] = min_
            x_axis[-1] = max_
            x_axis_list.append(np.round(x_axis, 10))
            neurons = np.zeros((self.n_presynaptic_neurons, len(x_axis)))

            for i in range(self.n_presynaptic_neurons):

                loc = (max_ - min_) * (i /(self.n_presynaptic_neurons-1)) + min_
                neurons[i] = norm.pdf(x_axis, loc, sigm[t])
                neurons[i] = neurons[i] / np.max(neurons[i])

            neurons_list.append(neurons)
            t += 1

        return neurons_list, x_axis_list


    def Lat_Spike(self):

        d = self.Gaussian_neurons()

        for i in range(len(self.df.columns)):

            k = len(self.df.iloc[:, i])
            st1 = np.tile(d[1][i], (k, 1))
            st2 = self.df.iloc[:, i].values.reshape(-1, 1)
            ind = (st1 == st2)
            exc = np.tile(d[0][i], (k, 1)).reshape(k, self.n_presynaptic_neurons, len(d[0][i][0]))[
                np.repeat(ind, self.n_presynaptic_neurons, axis=0).reshape(k, self.n_presynaptic_neurons, len(ind[0]))].reshape(k, self.n_presynaptic_neurons)
            lat_neuron = np.transpose(np.where(exc > 0.1, 1 - exc, np.nan))

            if i == 0:
                lat_neuron_total = lat_neuron
            else:
                lat_neuron_total = np.concatenate((lat_neuron_total, lat_neuron), axis = 0)

        lat_neuron_total[lat_neuron_total == 0] = 0.0001
        final_df = pd.DataFrame(lat_neuron_total)

        return final_df

    def model_data(self, ind, ind_type, lat_ne, start, end):

        train_stack = np.vstack((lat_ne[ind_type[ind, 0] + start:ind_type[ind, 0] + end],
                                lat_ne[ind_type[ind, 1] + start:ind_type[ind, 1] + end],
                                lat_ne[ind_type[ind, 2] + start:ind_type[ind, 2] + end]))
        train_stack = np.where(train_stack > 0, train_stack, 0)

        return train_stack

    def define_weights(self):
        Final_df = self.Lat_Spike()
        lat_ne = np.transpose(Final_df.values)
        ind_type = np.array(([0, 50, 100], [50, 100, 0], [100, 0, 50]))
        list_weight = np.zeros((3,40))

        for ind in range(3):

            train_stack = self.model_data(ind, ind_type, lat_ne, 0, 20)
            tr_ar = np.where(np.transpose(train_stack) > 0, 2 * (1 - np.transpose(train_stack)), 0)
            tr_ar[:, 20:] = tr_ar[:, 20:] * (-1)
            tr_ar = pd.DataFrame(tr_ar)
            tr_ar[20] = tr_ar.iloc[:,:20].sum(axis = 1) + 0.1
            tst_ar = np.float64(np.transpose(np.array(tr_ar.iloc[:,20:])))

            for i in range(1, len(tst_ar)):

                tst_ar[0][((np.round(tst_ar[0], 4) > 0.1) & (tst_ar[i] == 0))] += - np.float64(
                    np.sum(tst_ar[i][np.round(tst_ar[0], 4) > 0.1]) / len(tst_ar[0][((
                        np.round(tst_ar[0], 4) > 0.1) & (tst_ar[i] == 0))]))
                tst_ar[0][np.round(tst_ar[0], 4) > 0.1] += tst_ar[i][np.round(tst_ar[0], 4) > 0.1]
                tst_ar[0][tst_ar[0] < 0.1] = 0.1

            list_weight[ind, :] = tst_ar[0]

        return list_weight


    def LIF_SNN(self, n = 30, l = 60, v_spike = 100):
        Final_df = self.Lat_Spike()
        lat_ne = np.transpose(Final_df.values)
        ind_type = np.array(([0, 50, 100], [50, 100, 0], [100, 0, 50]))
        data = self.model_data(0, ind_type, lat_ne, 0, 20)
        weight = self.define_weights()

        V_min = 0
        V_spike = v_spike
        r = 5
        tau = 2.5
        dt = 0.01
        t_max = 10
        time_stamps = t_max / dt
        time_relax = 10
        v = np.zeros((n, l, int(time_stamps)))
        t_post = np.zeros((n, l))
        t_post_ = np.zeros((n, int(l / 3)))
        v[:, :, 0] = V_min

        for n in range(n):
            for u in range(l):

                t = 0
                f0 = (np.round(data[u][np.newaxis].T, 3) * 1000).astype(int)
                f1 = np.tile(np.arange(1000), (40, 1))
                f2 = np.where(((f1 == f0) & (f0 > 0)), 1, 0)
                f2 = f2 * weight[n][np.newaxis].T
                spike_list = np.sum(f2.copy(), axis = 0)

                for step in range(int(time_stamps) - 1):
                    if v[n, u, step] > V_spike:
                        t_post[n, u] = step
                        v[n, u, step] = 0
                        t = time_relax / dt
                    elif t > 0:
                        v[n, u, step] = 0
                        t = t - 1

                    v[n, u, step + 1] = v[n, u, step] + dt / tau * (-v[n, u, step] + r * spike_list[step])
            t_post_[n, :] = t_post[n, n * int(l / 3):n * int(l / 3) + int(l / 3)]

        return v, t_post_, t_post





if __name__ == '__main__':
    model = SNN_model('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv', 10)
    #d = model.Gaussian_neurons()
    fin = model.Lat_Spike()
    weights = model.define_weights()
    a = model.LIF_SNN()

    print(weights)
