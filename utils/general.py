import numpy as np
from numpy import polynomial as poly
from scipy import integrate
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt

def belleFigure(ax1, ax2, nfigure=None):
    fig = plt.figure(nfigure, constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(ax1, fontsize=24)
    ax.set_ylabel(ax2, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18, width=1)
    for tick in ax.xaxis.get_ticklabels():
        tick.set_weight('bold')
    for tick in ax.yaxis.get_ticklabels():
        tick.set_weight('bold')
    plt.tight_layout()
    return fig, ax
 
