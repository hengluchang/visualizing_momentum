import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class LossSurface:
    """A loss surface with L(x, y) = a * x ^2 + b * y ^2.

    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

        N = 1000
        x_list = np.linspace(-2.5, 2.5, N)
        y_list = np.linspace(-0.5, 0.5, N)
        self.X, self.Y = np.meshgrid(x_list, y_list)
        self.Z = self.a * (self.X ** 2) + self.b * (self.Y ** 2)

    def plot(self):
        fig, ax = plt.subplots()
        cmap = cm.get_cmap('Greens_r')
        cp = ax.contour(self.X, self.Y, self.Z, 50, cmap=cmap)
        cbar = fig.colorbar(cp)
        cbar.set_label('loss')

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.5, 0.5)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        return fig, ax

if __name__ == '__main__':
    a = 1 / 16
    b = 9

    loss_surface = LossSurface(a, b)
    fig, ax = loss_surface.plot()
    fig_name = 'loss_surface.png'
    fig.savefig(fig_name)

    print('{} saved.'.format(fig_name))