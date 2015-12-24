# -*- coding: utf-8 -*-

"""Copyright 2015 Jonas Leininger, Cologne, Germany.

Code supporting the jupyter notebook
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
from contextlib import contextmanager
from IPython.core.display import HTML
import matplotlib.pylab as pylab
#from book_style import *
import numpy as np
from scipy import signal, stats


x = np.linspace(0,1001,1000)
y = 22. + np.random.randn(1000)

def equal_axis():
    pylab.rcParams['figure.figsize'] = 10,10
    plt.axis('equal')


def reset_axis():
    pylab.rcParams['figure.figsize'] = 11, 4

def set_figsize(x=11, y=4):
    pylab.rcParams['figure.figsize'] = x, y

@contextmanager
def figsize(x=11, y=4):
    """
    Temporarily set the figure size using 'with figsize(a,b):'
    """

    size = pylab.rcParams['figure.figsize']
    set_figsize(x, y)
    yield
    pylab.rcParams['figure.figsize'] = size

def show_legend():
    #plt.legend(loc='upper center', bbox_to_anchor=(1, 0.5))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=2, mode="expand", borderaxespad=0.)


def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def plotHistogrammSchools(dictCount):
    with figsize(16,10):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.bar(range(len(dictCount)), dictCount.values(), color='g',align="center")
        ax1.set_xticks(range(len(dictCount)), list(dictCount.keys()))
        ax1.set_xticklabels(list(dictCount.keys()))

def plotHistogrammVisitors(dictCount):
    with figsize(16,10):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.bar(range(len(dictCount)), dictCount.values(), color='g',align="center")
        ax1.set_xticks(range(len(dictCount)), list(dictCount.keys()))
        ax1.set_xticklabels(list(dictCount.keys()))

def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def plot_different_gaussians():
    xs = np.linspace(-4,4,200)
    parameterList = [[0,1],[0,4],[0,0.5],[1,1]]
    labelList = [['$\mu$=0, $\sigma^2$=1','-'], ['$\mu$=0, $\sigma^2$=4',':'],
        ['$\mu$=0, $\sigma^2$=0.5','--'],['$\mu$=1, $\sigma^2$=1','-']]
    for mu, var in enumerate(parameterList):
        plt.plot(xs, stats.norm.pdf(xs, var[0], var[1]), ls=labelList[mu][1], label=labelList[mu][0])
    show_legend()
    plt.ylim(0,1)
    plt.savefig('differentGaussians.png',bbox_inches="tight")


def plot_allNokraOutDiffsHist(dataframe, nokraList, limits=[1.57,1.64]):
    axnumbers = len(nokraList)/3
    x_base = dataframe['regelwert'].values
    with figsize(12,8):
        fig = plt.figure()
        k = 1
        ax1 = fig.add_subplot(1,1,1)
        for k in range(1,len(nokraList)+1):
            x = dataframe[nokraList[k-1]].values
            ax1.hist(x - x_base,50, label='Nokra {!s}'.format(nokraList[k-1]),alpha=0.7)
            #ax1.set_xlim(limits[0],limits[1])
            show_legend()
        ax1.set_ylabel('count #')
        ax1.set_xlabel('Position x[m]')

def plot_forceVsGap(dataframe, limits1=[0,5000], limits2=[0,50]):
    arr = dataframe['x'].values
    with figsize(12,10):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(arr/1000, dataframe['H1_HGC_RolForActDS'])
        ax1.plot(arr/1000, dataframe['H1_HGC_RolForActNDS'])
        ax1.set_ylim(limits1[0],limits1[1])
        show_legend()
        ax1.set_ylabel('roll force [kN]')
        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(arr/1000, dataframe['H1_HGC_PosActDS'])
        ax2.plot(arr/1000, dataframe['H1_HGC_PosActNDS'])
        ax2.set_ylim(limits2[0],limits2[1])
        show_legend()
        ax2.set_xlabel('Position x[m]')
        ax2.set_ylabel('Gauge [mm]')
