'''
The Ising model
'''

import numpy as np


class Ising:
    def __init__(self, xlen, ylen, T, j=1, kb=1):
        '''
        Initilize the lattice and initial parameters 
        '''
        # Lattice
        self.xlen = xlen
        self.ylen = ylen
        self.n = xlen * ylen
        self.state = np.random.randint(0, 2, self.n).reshape(xlen, ylen)
        self.state[self.state == 0] = -1
        # Initial parameters
        self.j = j
        self.T = T
        self.kb = kb
        self.beta = 1 / (kb * T)

    def total_energy(self, state=None):
        if state is None:
            state = self.state
        # Works by going from top left to bottom right calculating energy bonds below and right
        spin_summation = 0
        for i in range(len(state)):
            for j in range(len(state[i])):
                for delta in ((1, 0), (0, 1)):
                    x_neigh, y_neigh = i + delta[0], j + delta[1]
                    # Edges of the lattice: periodic boundaries
                    if x_neigh < 0:
                        x_neigh = self.xlen - 1
                    if x_neigh >= self.xlen:
                        x_neigh = 0
                    if y_neigh < 0:
                        y_neigh = self.ylen - 1
                    if y_neigh >= self.ylen:
                        y_neigh = 0
                    spin_summation += state[i, j] * state[x_neigh, y_neigh]
        energy = - self.j * spin_summation
        return energy

    def total_magnetism(self, state=None):
        if state is None:
            state = self.state
        magnetism = 0
        for i in range(self.xlen):
            for j in range(self.ylen):
                magnetism += state[i, j]
        return magnetism

    def local_energy(self, spin, state=None):
        if state is None:
            state = self.state
        x, y = spin
        spin_summation = 0
        for move in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            x_neigh, y_neigh = x + move[0], y + move[1]
            # Edges of the lattice: periodic boundaries 
            if x_neigh < 0:
                x_neigh = self.xlen - 1
            if x_neigh >= self.xlen:
                x_neigh = 0
            if y_neigh < 0:
                y_neigh = self.ylen - 1
            if y_neigh >= self.ylen:
                y_neigh = 0
            spin_summation += state[x, y] * state[x_neigh, y_neigh]
        energy = - self.j * spin_summation
        return energy

    def metropolis(self, delta_energy):
        '''
        return: True for accepting new state, False for not accepting new state
        '''
        if delta_energy < 0:
            return True
        else:
            p = min(1, np.exp(-self.beta * delta_energy))
            r = np.random.random()
            if r <= p:
                return True
            else:
                return False

    def glauber(self):
        '''
        Glauber dynamics: updates state
        return: energy delta, magnetism delta
        '''
        rand_spin = np.random.randint(0, self.xlen), np.random.randint(0, self.ylen)
        orig_energy = self.local_energy(rand_spin)
        self.state[rand_spin[0], rand_spin[1]] *= -1
        new_energy = self.local_energy(rand_spin)
        delta_energy = new_energy - orig_energy

        # orig_state = state.copy()
        # new_state = state.copy()
        # random_spin = np.random.randint(0, self.xlen), np.random.randint(0, self.ylen)
        # new_state[random_spin[0], random_spin[1]] *= -1  # flips the particular spin (this is Glauber dynamics)
        # orig_energy = self.local_energy(orig_state, random_spin)
        # new_energy = self.local_energy(new_state, random_spin)
        # delta_energy = new_energy - orig_energy

        # Metropolis algorithm
        if self.metropolis(delta_energy):
            return delta_energy, 2 * self.state[rand_spin[0], rand_spin[1]] * (-1)  # abs(+1-(-1)) = 2
        else:
            # Go back to the previous state
            self.state[rand_spin[0], rand_spin[1]] *= -1
            return 0, 0

    def kawasaki(self):
        '''
        Kawasaki dynamics
        return: next state derermined by kawasaki dynamics, energy delta, magnetism delta
        '''

        # Find two random spins that are different
        rand_spin_1 = np.random.randint(0, self.xlen), np.random.randint(0, self.ylen)
        rand_spin_2 = np.random.randint(0, self.xlen), np.random.randint(0, self.ylen)
        while self.state[rand_spin_1[0], rand_spin_1[1]] == self.state[rand_spin_2[0], rand_spin_2[1]]:
            rand_spin_1 = np.random.randint(0, self.xlen), np.random.randint(0, self.ylen)
            rand_spin_2 = np.random.randint(0, self.xlen), np.random.randint(0, self.ylen)

        orig_energy_1 = self.local_energy(rand_spin_1)
        orig_energy_2 = self.local_energy(rand_spin_2)
        temp = self.state[rand_spin_1[0], rand_spin_1[1]]
        self.state[rand_spin_1[0], rand_spin_1[1]] = self.state[rand_spin_2[0], rand_spin_2[1]]
        self.state[rand_spin_2[0], rand_spin_2[1]] = temp
        new_energy_1 = self.local_energy(rand_spin_1)
        new_energy_2 = self.local_energy(rand_spin_2)

        correction = 0
        if rand_spin_1[0] - rand_spin_2[0] == 0:
            if abs(rand_spin_1[1] - rand_spin_2[1]) == 1:
                correction = -self.j
        elif rand_spin_1[1] - rand_spin_2[1] == 0:
            if abs(rand_spin_1[0] - rand_spin_2[0]) == 1:
                correction = -self.j
        orig_energy = orig_energy_1 + orig_energy_2 + correction
        new_energy = new_energy_1 + new_energy_2 + correction
        delta_energy = new_energy - orig_energy

        # Metropolis
        if self.metropolis(delta_energy):
            return delta_energy, 0
        else:
            # Go back to the previous state
            temp = self.state[rand_spin_1[0], rand_spin_1[1]]
            self.state[rand_spin_1[0], rand_spin_1[1]] = self.state[rand_spin_2[0], rand_spin_2[1]]
            self.state[rand_spin_2[0], rand_spin_2[1]] = temp
            return 0, 0

    def run(self, steps, dynamics, record_freq=1):
        '''
        steps: int, number of steps simulation takes
        dynamics: 'glauber' or 'kawasaki' dynamics
        '''
        # Record the magnetisation and energy
        # Calculate them efficiently by adding up the change to the original values
        energy = self.total_energy(self.state)
        magnetism = self.total_magnetism(self.state)
        
        energy_data = [energy]
        magnetism_data = [magnetism]

        for i in range(steps):
            if dynamics == 'glauber':
                delta_energy, delta_magnetism = self.glauber()
            elif dynamics == 'kawasaki':
                delta_energy, delta_magnetism = self.kawasaki()
            else:
                raise Exception('Unknown dynamic (try kawasaki or glauber)')

            energy += delta_energy
            magnetism += delta_magnetism
            if i % record_freq == 0:
                energy_data.append(energy)
                magnetism_data.append(magnetism)

        return self.state, (energy_data, magnetism_data)


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    T = 1
    # Initial state of the lattice
    x = 50
    y = 50
    ising = Ising(xlen=x, ylen=y, T=T)
    steps = 100000
    dynamics = 'kawasaki'
    initial = ising.state

    _, data = ising.run(steps, dynamics)
    final = ising.state

    energy_data, magnetism_data = data



