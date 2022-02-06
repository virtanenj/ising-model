'''
Visualisation of the Ising model
'''

from ising import Ising
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation as manimation
from threading import Thread


class isingView():
    def __init__(self, model, steps, dynamics, printFreq):
        self.model = model
        self.steps = steps
        self.dynamics = dynamics
        self.printFreq = printFreq  # make use of this

        self.fig, self.ax = plt.subplots()

        cmap = matplotlib.colors.ListedColormap(['grey', 'white'])

        self.im = self.ax.imshow(self.model.state, cmap=cmap)
        self.anim = None

    def run(self, interval):
        # # What's this?
        thread = Thread(target=self.model.run,
                        args=(self.steps, self.dynamics,))
        thread.start()

        self.ani = manimation.FuncAnimation(self.fig, self.animate, interval=interval, blit=True)
        plt.show()

    def animate(self, frame):
        self.im.set_data(self.model.state)
        return self.im,


if __name__ == "__main__":
    xlen = 50
    ylen = 50
    T = 2
    model = Ising(xlen, ylen, T)
    steps = 100000
    dynamics = 'glauber'
    printFreq = 10
    view = isingView(model, steps, dynamics, printFreq)
    frames = 1000
    view.run(interval=10)
