#! /anaconda3/envs/ot_env

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

n = 100  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a = gauss(n, m=20, s=5)
b = gauss(n, m=50, s=10)

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
M /= M.max()

# change SD - sinkhorn
fig = plt.figure(3, figsize=(5, 5))
def update_sd(sd):
    a = gauss(n, m=40, s=sd) 
    b = gauss(n, m=50, s=10)
    Gs = ot.sinkhorn(a, b, M, 1e-2)
    pl = ot.plot.plot1D_mat(a, b, Gs, 'OT matrix G0')
    return pl,
    
ani = FuncAnimation(fig, update_sd, frames=np.arange(1, 20, 0.5), blit=False, interval=50)
ani.save('ot_1d.gif', fps=10)
plt.show()