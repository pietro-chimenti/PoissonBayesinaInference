#!/usr/bin/env python
"""This files provides basic tools to make bayesian inferences on simple poissonian models.
"""
import warnings
import numpy as np
from scipy.stats import poisson
from scipy.stats import gamma
import matplotlib.pyplot as plt
import math
import emcee



class SignalAndNoise :
    """This class represent a simple poissonian model for bayesian analysis"""

    ndim, nwalkers = 1, 100 
    big_integer = 100


    def log_like(self, mu, data):
        """ This function represent the logarithm of the poissonian likelihood"""
        if mu > 0 :
            if data < self.big_integer : 
                return data*np.log(mu)-mu-np.log(float(math.factorial(data)))
            else : # use stirling! Need to improve terms.
                return data*np.log(mu)-mu-data*np.log(data)+data
        else: return -1*math.inf

    def log_prior_uniform(self, mu):
        if mu <= 0: return -1.*math.inf
        return (self.r-1)*mu-self.v*mu

    def log_prob(self, mu):
        lp = self.log_prior(mu)
        if lp == -1.*math.inf: return -1.*math.inf, 0, 0, 0, 0
        ll = self.log_like(mu, self.ov)
        if ll == -1.*math.inf: return -1.*math.inf, 0, 0, 0, 0
        posterior_sample = poisson.rvs(mu)
        psl = self.log_like(mu, posterior_sample)
        return ll+lp, ll, lp, posterior_sample, psl

    def __init__(self, observed_value, prior = 'uniform', mean = 0, std = 0, *args):
        self.ov = observed_value

        if prior == 'uniform':
            self.log_prior = self.log_prior_uniform
        if prior == 'jeffreys':
            if mean == 0 or std == 0: raise ValueError('Prior mean and std must be positive!')
            self.r = mean**2/std**2
            self.v = mean/std**2
            self.log_prior = self.log_prior_gamma

        self.mu_min = round(self.ov-5*math.sqrt(self.ov))
        self.mu_max = round(self.ov+5*math.sqrt(self.ov))
        if self.mu_min < 0: self.mu_min = 0
        if self.ov == 0 :
            self.mu_min = 0
            self.mu_max = 5

        self.samples_list = []
        self.blobs_list   = []
        self.acceptance_fraction_list = []
        self.autocorr_time_list = []

    def run(self, samples = 1000, seed = 42):
        if self.samples_list:
            warnings.warn("Sampler already run!")
        else:
            np.random.seed(seed)
            for i in range(8):
                print("Running chain n.",i)
                p0 = self.mu_max*np.absolute(np.random.randn(self.nwalkers, self.ndim))
                sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob)
                sampler.run_mcmc(p0, samples)
                self.samples_list.append(sampler.get_chain(flat=True))
                self.blobs_list.append(sampler.get_blobs())
                self.acceptance_fraction_list.append(sampler.acceptance_fraction)
                self.autocorr_time_list.append(sampler.get_autocorr_time())
            self.diagnose()

    def diagnose(self):
        for i in range(8):
            print("Chain ",i)
            print( "Mean acceptance fraction: {0:.3f}".format(np.mean(self.acceptance_fraction_list[i])))
            print( "Mean autocorrelation time: {0:.3f} steps".format(np.mean(self.autocorr_time_list[i])))