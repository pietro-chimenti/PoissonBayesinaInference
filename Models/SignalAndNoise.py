#!/usr/bin/env python
"""This files provides basic tools to make bayesian inferences on simple poissonian models.
"""


import math 
import numpy as np
from scipy.stats import gamma 

class SignalAndNoise :
    """This class represent a double parameter poissonian model for bayesian analysis using
    Monte Carlo Markov Chain"""
    

    '''PRIOR OFF:'''
        
    def log_prior_off_uniform(self, mu):
        if mu > 0:
            return 0
        else:
            return - math.inf
    
    def log_prior_off_jeffrey(self, mu):
        if mu > 0:
            return 0.5*np.log(mu)
        else:
            return -math.inf
        
    def log_prior_off_gamma(self,mu, alpha, beta):
        if mu > 0:
            self.gamma = gamma.pdf(mu, alpha, scale = 1/beta)
            return np.log(self.gamma)
        else:
            return -math.inf 
        
    '''PRIOR OFF+ON'''
        
    def log_prior_off_on_uniform(self, mu_on, mu_off):
        if mu_off > 0 and mu_on > 0:
            return 0
        else:
            return - math.inf
        
    def log_prior_off_on_jeffrey(self, mu_on, mu_off):
           if mu_off > 0 and mu_on > 0:
               return 0.5*np.log(mu_on + mu_off)
           else:
               return -math.inf
            
    def log_prior_off_on_gamma(self,mu_on, mu_off, alpha, beta):
        if mu_off > 0 and mu_on > 0:
            self.gamma = gamma.pdf(mu_on + mu_off, alpha, scale = 1/beta)
            return np.log(self.gamma)
        else:
            return -math.inf 
        
    '''LIKELYHOOD'''
    
    def log_like_off(self,mu,data):
        if mu > 0:
            if data <= 20 and data >0 :
                return -mu + data*np.log(mu) - np.log(math.factorial(data))
            elif data > 20: #aprox de stirling
                return -mu+data*np.log(mu)-data*np.log(data)+data-0.5*np.log(2*data*math.pi)
            else:
                return print('Data must be an positive integer!')
        else: 
            return - math.inf
        
    def log_like_off_on(self,mu_on, mu_off ,data):
        self.mu_sum = mu_on + mu_off
        if mu_on > 0 and mu_off > 0:
            if data <= 20 and data >0 :
                return - self.mu_sum + data*np.log(self.mu_sum) - np.log(math.factorial(data))
            elif data > 20: #aprox de stirling
                return -self.mu_sum+data*np.log(self.mu_sum)-data*np.log(data)+data-0.5*np.log(2*data*math.pi)
            else:
                return print('Data must be an positive integer!')
        else: 
            return - math.inf        
    
        
    def __init__(self, observed_value):
        
        print('it´s working!')

    
    
