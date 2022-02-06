'''
Susceptibility
chi = 1/(N kb T) (Mean(M^2) - Mean(M)^2)
where the total magnetisation 
M = sum_i S_i

Plot chi as a funciton of T. Use this to estimate the critical temperature T_c. Have 0 <= T <= 3 with delta T = 0.1

Note: 1 sweep = 2500 (= 50*50) attempted flips

To do:
- Data saving to external files (like .dat, .cvs or even .txt)
- Error bars
- Critical temperature estimation
- Analysis with Kawasaki dynamics
'''

from ising import Ising
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


def chi(M_data, N, kb, T):
    varM = (np.array(M_data)**2).mean() - np.array(M_data).mean()**2
    return 1 / (N * kb * T) * varM


def scaled_heat_capacity(E_data, N, kb, T):
    varE = (np.array(M_data)**2).mean() - np.array(M_data).mean()**2
    return 1 / (N * kb * T**2) * varE


Ts = np.arange(1, 3 + 0.1, 0.1)  # ideally np.arange(1, 3 + 0.1, 0.1)
runs = len(Ts)

xlen = 50
ylen = 50
sweep = xlen * ylen  # 2500
nsweeps = 1000  # ideally 10000
steps = sweep * nsweeps
dynamics = 'glauber'
record_freq = 10 * sweep

dataset = []
for i in range(runs):
    T = Ts[i]
    # Check if data already exists for this T, nsweeps, (anything else?)
    # data.csv has columns for nsweeps, T, state_data (maybe?), energy_data, magnetism_data
    if os.path.isfile('./data.csv'):
        pass
    else:
        pass

    with open('./data.csv', 'r') as f:
        pass

    model = Ising(xlen, ylen, T)
    _, (state_data, energy_data, magnetism_data) = model.run(steps, dynamics, record_freq)

    # Recording has been done for every 10 sweep.
    # Skip the first ten of these (over all 100 first sweep) to only sample the settled values
    magnetism_data = magnetism_data[10:]
    energy_data = energy_data[10:]

    # Does the data need any further sampling?

    dataset.append((magnetism_data, energy_data, T))

    print(str(i + 1) + '/' + str(runs))


N = xlen * ylen
kb = 1
chi_data = []
c_data = []
for i in range(len(dataset)):
    M_data, E_data, T = dataset[i]
    chi_data.append(chi(M_data, N, kb, T))
    c_data.append(scaled_heat_capacity(E_data, N, kb, T))

plt.plot(Ts, chi_data, '-o')
plt.xlabel('T')
plt.ylabel('chi(T)')
plt.grid()
plt.show()
