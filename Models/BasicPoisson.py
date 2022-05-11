#!/usr/bin/env python
"""This files provides basic tools to make bayesian inferences on simple poissonian models.
"""

import numpy as np
from scipy.stats import poisson
from scipy.stats import gamma
import math


class BasicPoisson :
    """This class represent a simple poissonian model for bayesian analysis"""
    
    def log_like(mu, *data):
        """ This function represent the logarithm of the poissonian likelihood"""
        if (mu > 0):
            return np.log(gamma.pdf(mu,data[0]+1))
        else: return -1*math.inf



