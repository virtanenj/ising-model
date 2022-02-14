'''
Setting up and running the Ising model for T values in [1,3] with either glauber or kawasaki dynamics.

Note: 
- 1 sweep = 2500 (= 50*50) attempted flips
- 100 sweeps at the start and 10 sweep afterwards is enough to avoid correlation
'''

from ising import Ising
import numpy as np
import matplotlib.pyplot as plt
import csv


# Experiment parameters:
Ts = np.arange(1, 3, 0.1)  # ideally np.arange(1, 3 + 0.1, 0.1)
runs = len(Ts)

# ideally 50x50
xlen = 50
ylen = 50
sweep = xlen * ylen  # 2500
nsweeps = 10000  # ideally 10000
steps = sweep * nsweeps
dynamics = 'kawasaki'
record_freq = 10 * sweep

init_state = None
if dynamics == 'glauber':
    # Initialize state for all +1 (only for glauber)
    init_state = np.ones(xlen * ylen).reshape(xlen, ylen)
elif dynamics == 'kawasaki':
    # Initialize state for half +1 and half -1
    init_state = np.ones(xlen * ylen)
    init_state[:int((xlen * ylen) / 2)] = init_state[:int((xlen * ylen) / 2)] * (-1)
    init_state = init_state.reshape(xlen, ylen)

# Data:
header = ['T', 'energy data', 'magnetism data']
myfile = open('data50x50kawasaki_energy-calc-correction_10000nsweeps.csv', 'w')
writer = csv.writer(myfile)
writer.writerow(header)

for i in range(runs):
    T = Ts[i]
    model = Ising(xlen, ylen, T)
    model.state = init_state
    last_state, (energy_data, magnetism_data) = model.run(steps, dynamics, record_freq)
    init_state = last_state

    # Recording has been done for every 10 sweep.
    # Skip the first 10 recording points (= 100 first sweep)
    magnetism_data = magnetism_data[10:]
    energy_data = energy_data[10:]

    writer.writerow([T, energy_data, magnetism_data])

    print(str(i + 1) + '/' + str(runs))

myfile.close()


if __name__ == '__main__':
    pass
