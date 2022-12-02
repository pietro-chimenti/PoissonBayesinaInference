#!/usr/bin/env python
"""This module tests the bayesian inference of the "basic" poissonian model.
"""

import sys
import numpy as np
from scipy.stats import poisson
from scipy.stats import gamma
import matplotlib.pyplot as plt
import math
import emcee

from Models import BasicPoisson as BP

def main():    
    print("Input the observed value:")
    ov_string = input()
    try:
        ov = int(ov_string)
    except ValueError:
        print("Value Error: must be an integer!")
        sys.exit(0)    
    if ov < 0 :
        print("Observed value must not be negative!")
        sys.exit(0)


    mu_min = round(ov-5*math.sqrt(ov))
    mu_max = round(ov+5*math.sqrt(ov))
    if ov == 0 :
        mu_min = 0
        mu_max = 5

    print("Running MCMC...")
    bp_model = BP.BasicPoisson()
    
    ndim, nwalkers = 1, 100 
    p0 = mu_max*np.absolute(np.random.randn(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, 1, bp_model.log_like, args=([ov]))
    sampler.run_mcmc(p0, 1000)
    samples = sampler.get_chain(flat=True)
    print("done!")
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(samples[:, 0], label="Dados Simulação")
    plt.title("Simulação Cadeia de Markov")
    ax.legend(loc='best', frameon=False)
    plt.grid(axis="x", linestyle="-.")
    plt.axvline(x=20000, color="r", label="Corte de Dados")
    plt.legend()
    plt.xlabel("numero de amostra")
    plt.ylabel(r'$\lambda$')
    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    value, bins, _ = ax.hist(samples[20000:, 0], 1000, color="k", histtype="step", density=True,
    label="Dados Simulação")
    posterior = gamma.pdf(bins,ov+1)
    plt.plot(bins,posterior, label="Solução analítica")
    plt.title("Distribuição de Poisson")
    ax.legend(loc='best', frameon=False)
    plt.grid(linestyle="-.")
    plt.legend()
    plt.xlabel(r'$\lambda$')
    plt.ylabel("p.d.f.")
    plt.show()


if __name__ == "__main__":
    main()
