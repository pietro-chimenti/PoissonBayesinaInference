#!/usr/bin/env python
"""This module produces plots for the bayesian inference of the poissonian model.
"""

import numpy as np
from scipy.stats import poisson
from scipy.stats import gamma
import matplotlib.pyplot as plt
import math
import emcee

def main():

    print("First plot poisson distribution")
    fig, ax = plt.subplots(1, 1)
    mu_1 = 0.6
    x_1 = np.arange(poisson.ppf(0.01, mu_1),
        poisson.ppf(0.99, mu_1))
    ax.plot(x_1, poisson.pmf(x_1, mu_1), 'bo', ms=8, label='poisson pmf - 0.6')

    mu_2 = 3.0
    x_2 = np.arange(poisson.ppf(0.01, mu_2),
        poisson.ppf(0.99, mu_2))
    ax.plot(x_2, poisson.pmf(x_2, mu_2), 'ro', ms=8, label='poisson pmf - 3.0')

    ax.legend(loc='best', frameon=False)
    plt.show()

    print("Assume we observe 0 or 3: plot likelihood")

    fig, ax = plt.subplots(1, 1)
    mu_interval = np.arange(0, 10, 0.01)
    ax.plot( mu_interval, poisson.pmf(0,mu_interval), label='k=0' )
    ax.plot( mu_interval, poisson.pmf(3,mu_interval), label='k=3' )
    ax.plot( mu_interval, gamma.pdf(mu_interval ,3+1), label='gamma - a = 3+1')
    ax.legend(loc='best', frameon=False)
    ax.set_title('Likelihood')
    plt.show()

    print("Now let's draw random samples ")

    def log_prob(mu):
        if (mu > 0):
            return np.log(gamma.pdf(mu,4))
        else: return -1*math.inf

    ndim, nwalkers = 1, 100
    a = [3+1]
    #p0 = np.ones((100,1)) 
    p0 = np.absolute(np.random.randn(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, 1, log_prob)
    sampler.run_mcmc(p0, 1000)
    samples = sampler.get_chain(flat=True)
    fig, ax = plt.subplots(1, 1)
    ax.hist(samples[1000:, 0], 100, color="k", histtype="step")
    ax.plot( mu_interval, 100*150*gamma.pdf(mu_interval ,3+1), label='gamma - a = 3+1')
    ax.legend(loc='best', frameon=False)
    plt.show()


if __name__ == "__main__":
    main()
