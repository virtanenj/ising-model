import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


class Ising_model:
    def __init__(self, xlen, ylen, j, T, kb=1):
        '''
        Initilize the lattice and initial parameters 
        '''
        # Lattice
        self.xlen = xlen
        self.ylen = ylen
        self.n = x * y
        self.state = np.random.randint(0, 2, self.n).reshape(x, y)
        self.state[self.state == 0] = -1
        # Initial parameters
        self.j = j
        self.T = T
        self.kb = kb
        self.beta = 1 / (kb * T)

    def total_energy(self, state=None):
        if state is None:
            state = self.state
        # works by going from top left to bottom right calculating paindings between below and right
        spin_summation = 0
        for i in range(len(state)):
            for j in range(len(state[i])):
                # neighbour
                for delta in ((1, 0), (0, 1)):
                    neigh = i + delta[0], j + delta[1]
                    # edges of the lattice
                    if neigh[0] >= 0 and neigh[1] >= 0 and neigh[0] < self.xlen and neigh[1] < self.ylen:
                        spin_summation += state[i, j] * state[neigh[0], neigh[1]]
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

    def local_energy(self, state, spin):
        x, y = spin
        spin_summation = 0
        for move in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            x_neigh, y_neigh = x + move[0], y + move[1]
            # Consider periodic boundaries 
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

    def glauber(self, state):
        '''
        Glauber dynamics
        return: next state determined by glauber dynamics, energy delta, magnetism delta
        '''
        orig_state = state.copy()
        new_state = state.copy()
        random_spin = np.random.randint(0, self.xlen), np.random.randint(0, self.ylen)
        new_state[random_spin[0], random_spin[1]] *= -1  # flips the particular spin (this is Glauber dynamics)
        orig_energy = self.local_energy(orig_state, random_spin)
        new_energy = self.local_energy(new_state, random_spin)
        delta_energy = new_energy - orig_energy

        # Metropolis algorithm
        if self.metropolis(delta_energy):
            return new_state, delta_energy, state[random_spin[0], random_spin[1]] * (-1)
        else:
            return orig_state, 0, 0

    def kawasaki(self, state):
        '''
        Kawasaki dynamics
        return: next state derermined by kawasaki dynamics, energy delta, magnetism delta
        '''
        orig_state = state.copy()
        new_state = state.copy()
        # Find two random spins that are different (is this smart to do?)
        random_spin_1 = np.random.randint(0, self.xlen), np.random.randint(0, self.ylen)
        random_spin_2 = np.random.randint(0, self.xlen), np.random.randint(0, self.ylen)
        while state[random_spin_1[0], random_spin_1[1]] == state[random_spin_2[0], random_spin_2[1]]:
            random_spin_1 = np.random.randint(0, self.xlen), np.random.randint(0, self.ylen)
            random_spin_2 = np.random.randint(0, self.xlen), np.random.randint(0, self.ylen)

        new_state[random_spin_1[0], random_spin_1[1]] = state[random_spin_2[0], random_spin_2[1]]
        new_state[random_spin_2[0], random_spin_2[1]] = state[random_spin_1[0], random_spin_1[1]]

        # orig_energy = self.local_energy(orig_state, random_spin_1) + self.local_energy(orig_state, random_spin_2)
        # new_energy = self.local_energy(new_state, random_spin_1) + self.local_energy(new_state, random_spin_2)

        orig_energy_1 = self.local_energy(orig_state, random_spin_1)
        new_energy_1 = self.local_energy(new_state, random_spin_1)
        delta_energy_1 = new_energy_1 - orig_energy_1
        orig_energy_2 = self.local_energy(orig_state, random_spin_2)
        new_energy_2 = self.local_energy(new_state, random_spin_2)
        delta_energy_2 = new_energy_2 - orig_energy_2
        # what if local region of swapped spins overlap?
        correction = 0
        if random_spin_1[0] - random_spin_2[0] == 0:
            if abs(random_spin_1[1] - random_spin_2[1]) == 1:
                correction = 2 * self.j
        elif random_spin_1[1] - random_spin_2[1] == 0:
            if abs(random_spin_1[0] - random_spin_2[0]) == 1:
                correction = 2 * self.j
        delta_energy = delta_energy_1 + delta_energy_2 + correction

        # Metropolis
        if self.metropolis(delta_energy):
            return new_state, delta_energy, 0  # In kawasaki dynamics delta magnetism = 0 (correct?)
        else:
            return orig_state, 0, 0

    def simulation(self, steps, dynamics, record=False):
        '''
        steps: int, number of steps simulation takes
        dynamics: 'glauber' or 'kawasaki' dynamics
        '''
        # Record the magnetisation and energy here
        # Calculate them efficiently by adding up the change to the original values
        state = self.state.copy()
        energy = self.total_energy(state)
        magnetism = self.total_magnetism(state)
        
        if record:
            state_data = [state]
            energy_data = [energy]
            magnetism_data = [magnetism]

        for _ in range(steps):
            if dynamics == 'glauber':
                state, delta_energy, delta_magnetism = self.glauber(state)
            elif dynamics == 'kawasaki':
                state, delta_energy, delta_magnetism = self.kawasaki(state)
            else:
                raise Exception('Unknown dynamic (try  kawasaki or glauber)')
            energy += delta_energy
            magnetism += delta_magnetism

            if record:
                state_data.append(state)
                energy_data.append(energy)
                magnetism_data.append(magnetism)

        self.state = state
        return state, (state_data, energy_data, magnetism_data)


T = 1
kb = 1
beta = 1 / (T * kb)
J = 1

# Initial state of the lattice
x = 50
y = 50
ising = Ising_model(xlen=x, ylen=y, j=J, T=T, kb=kb)
steps = 1000
dynamics = 'glauber'
initial = ising.state
_, data = ising.simulation(steps, dynamics, record=True)
final = ising.state

state_data, energy_data, magnetism_data = data

def animation(anim_data, frames):
    def anim_func(t):
        shot = anim_data[t]
        im.set_array(shot)
        return [im]
    fig = plt.figure()
    init_shot = anim_data[0]
    im = plt.imshow(init_shot, interpolation='none', aspect='auto', vmin=0, vmax=1)
    im.set_data(init_shot)
    anim = manimation.FuncAnimation(fig, anim_func, frames=frames, interval=0.1)
    anim.save('anim.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

# Visualization

# Animations
# Why is is so slow???
animation(data[0], frames=steps)

# Figures
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(initial)
ax1.set_title('Initial lattice')

ax2.imshow(final)
ax2.set_title('Final lattice')
plt.savefig('fig.jpg')
# plt.show()
