import collections

import matplotlib.pyplot as plt

from loss_surface import LossSurface


class Momentum:
    """ Momentum sgd update and plot weight trajectory.

    """
    def __init__(self, lr, iteration, beta, loss_surface):
        self.lr = lr
        self.iteration = iteration
        self.beta = beta
        self.a = loss_surface.a
        self.b = loss_surface.b

        self.w_1, self.w_2, self.v_1, self.v_2 = self.init_parameters()

    def run(self):
        w_1_curve = list()
        w_2_curve = list()
        v_1_curve = list()
        v_2_curve = list()

        w_1_curve.append(self.w_1)
        w_2_curve.append(self.w_2)
        v_1_curve.append(self.v_1)
        v_2_curve.append(self.v_2)

        for step in range(self.iteration):
            self.gd_update()
            loss = self.calculate_loss()

            w_1_curve.append(self.w_1)
            w_2_curve.append(self.w_2)
            v_1_curve.append(self.v_1)
            v_2_curve.append(self.v_2)

            print('iteration: {}, w1: {}, w2: {}, v1: {}, v2:{}, loss: {}'.format(
                  step+1, self.w_1, self.w_2, self.v_1, self.v_2, loss))

        return w_1_curve, w_2_curve, v_1_curve, v_2_curve

    def gd_update(self):
        self.v_1 = self.beta * self.v_1 + 2 * self.a * self.w_1
        self.w_1 -= self.lr * self.v_1

        self.v_2 = self.beta * self.v_2 + 2 * self.b * self.w_2
        self.w_2 -= self.lr * self.v_2

    def calculate_loss(self):
        loss = self.a * (self.w_1**2) + self.b * self.w_2**2

        return loss

    def save_plot(self, loss_surface_fig, loss_surface_ax, w_1_curve, w_2_curve):
        loss_surface_ax.plot(w_1_curve, w_2_curve, color="black")

        fig_name = 'sgd_beta_{}_lr_{}_iter_{}_with_contour.png'.format(self.beta, self.lr, self.iteration)
        loss_surface_fig.savefig(fig_name)

        print('{} saved.'.format(fig_name))

    @staticmethod
    def init_parameters():
        w_1 = -2.4
        w_2 = 0.2
        v_1 = 0
        v_2 = 0

        return w_1, w_2, v_1, v_2


def save_velocity_plot(name, v_curves):
    fig, ax = plt.subplots()
    ax.plot(v_curves['beta_0.0'], label='beta=0.0')
    ax.plot(v_curves['beta_0.8'], label='beta=0.8')
    ax.plot(v_curves['beta_0.9'], label='beta=0.9')
    ax.legend()

    ax.set_xlabel('iterations')
    ax.set_ylabel('velocity')

    fig_name = 'velocity_{}.png'.format(name)
    fig.savefig(fig_name)

    print('{} saved.'.format(fig_name))


def main():
    lr = 0.1
    iteration = 50

    v_1_curves = collections.defaultdict(list)
    v_2_curves = collections.defaultdict(list)

    for beta in [0.0, 0.8, 0.9]:
        a = 1 / 16
        b = 9
        ls = LossSurface(a, b)
        ls_fig, ls_ax = ls.plot()

        demo = Momentum(lr=lr, iteration=iteration, beta=beta, loss_surface=ls)
        w_1_curve, w_2_curve, v_1_curve, v_2_curve = demo.run()
        demo.save_plot(ls_fig, ls_ax, w_1_curve, w_2_curve)

        v_1_curves['beta_{}'.format(beta)] = v_1_curve
        v_2_curves['beta_{}'.format(beta)] = v_2_curve

    save_velocity_plot('v_1', v_1_curves)
    save_velocity_plot('v_2', v_2_curves)

if __name__ == '__main__':
    main()
