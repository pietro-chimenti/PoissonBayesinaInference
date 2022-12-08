#!/usr/bin/env python
"""This files provides basic tools to make bayesian inferences on simple poissonian models.
"""

import numpy as np
from scipy.stats import poisson
from scipy.stats import gamma
import math


class BasicPoisson :
    """This class represent a simple poissonian model for bayesian analysis"""
    
    def log_like(self, mu, *data):
        """ This function represent the logarithm of the poissonian likelihood"""
        if mu > 0 :
            if data[0] < 100 : 
                return data[0]*np.log(mu)-mu-np.log(float(math.factorial(data[0])))
            else : # use stirling! Need to improve terms.
                return data[0]*np.log(mu)-mu-data[0]*np.log(data[0])+data[0]
        else: return -1*math.inf

    def log_prior_uniform(self, mu):
        if mu <= 0: return -1*math.inf
        return 0

    def log_prior_jeffreys(self, mu):
        if mu <= 0: return -1.*math.inf
        return -0.5*mu
    
    def log_prior_gamma(self, mu):
        if mu <= 0: return -1.*math.inf
        return (self.r-1)*mu-self.v*mu
    
    def log_prob(self, mu, *data):
        lp = self.log_prior(mu)
        if lp == -1.*math.inf: return -1.*math.inf
        ll = self.log_like(mu, *data)
        if ll == -1.*math.inf: return -1.*math.inf
        return ll+lp
    
    def __init__(self, prior = 'uniform', mean = 0, std = 0, *args):
        if prior == 'uniform':
            self.log_prior = self.log_prior_uniform
        if prior == 'jeffreys':
            self.log_prior = self.log_prior_jeffreys 
        if prior == 'gamma':
            if mean == 0 or std == 0: raise ValueError('Prior mean and std must be positive!')
            self.r = mean**2/std**2
            self.v = mean/std**2
            self.log_prior = self.log_prior_gamma