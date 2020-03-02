#! /anaconda3/envs/ot_env

import numpy as np
import matplotlib.pylab as pl
import ot
from ot.datasets import make_1D_gauss as gauss
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

n = 100  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a = gauss(n, m=20, s=5)  # m= mean, s= std
b = gauss(n, m=50, s=10)

# loss matrix
M = ot.utils.dist0(n)
M /= M.max()

A = np.vstack((a, b)).T
n_distr = A.shape[1]


# plot l2 barycenter
fig, ax = plt.subplots(figsize=(10,5))
xdata, ydata = x, []
for i in range(n_distr):
    plt.plot(x, A[:, i], '--')
ln1, = plt.plot([], [], 'r')
alpha_text = ax.text(0.5, 0.98, 'blah', fontsize=10, c='r',
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes)

def update_l2(a):
    weights = np.array([1 - a, a])
    bary_l2 = A.dot(weights)
    ln1.set_data(xdata, bary_l2)
    alpha_text.set_text('alpha = ' + str(np.round(a, 2)))
    return ln1, alpha_text,

ani_l2 = FuncAnimation(fig, update_l2, frames=np.linspace(0.01, 1.0, 50), interval=50, repeat=True)
plt.title('L2 Barycenter Sliding')
plt.legend(['Distr. 1', 'Distr. 2', 'L2 Barycenter'])
ani_l2.save('l2_barycenter.gif', fps=10)
plt.show()

# plot wasserstein barycenter
fig, ax = plt.subplots(figsize=(10,5))
reg = 1e-3
xdata, ydata = x, []

for i in range(n_distr):
    plt.plot(x, A[:, i], '--')
ln2, = plt.plot([], [], 'g')
alpha_text = ax.text(0.5, 0.98, '', fontsize=10, c='g',
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes)

def update_wass(a):
    weights = np.array([1 - a, a])
    bary_wass = ot.bregman.barycenter_sinkhorn(A, M, reg, weights)
    ln2.set_data(xdata, bary_wass)
    alpha_text.set_text('alpha = ' + str(np.round(a, 2)))
    return ln2, alpha_text,

ani_wass = FuncAnimation(fig, update_wass, frames=np.linspace(0.01, 1.0, 50), interval=50, repeat=True)
plt.title('Wasserstein Barycenter Sliding')
plt.legend(['Distr. 1', 'Distr. 2', 'Wasserstein Barycenter'])
ani_wass.save('wass_barycenter.gif', fps=10)
plt.show()