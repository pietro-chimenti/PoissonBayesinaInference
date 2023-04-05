#!/usr/bin/env python
"""This files provides basic tools to make bayesian inferences on simple poissonian models.
"""

import sys
import numpy as np
from scipy.stats import poisson
from scipy.stats import gamma 
import pandas as pd


class BasicPoisson :
    """This class represent a simple poissonian model for bayesian analysis"""
    
    ''' Update Parameters '''
    def gamma_change(self,data, shape, scale):
        r = np.sum(data) + shape
        v = len(data) + scale
        return r, v
    
    '''Record the variables'''
    def __init__(self, observed_value, events_number, prior = 'uniform', mean = 0, std = 0, *args):
        
        # Sampler
        self.ov = observed_value
        self.ev = events_number
        
        self.data = poisson.rvs(mu = self.ov, size = self.ev)
        
        # New Parameters
        if prior == 'uniform':
            self.r , self.v = self.gamma_change(data = self.data, shape= 0.5, scale = 0)
        if prior == 'jeffreys':
            self.r , self.v = self.gamma_change(data = self.data, shape= 1, scale = 0)
        if prior == 'gamma':
            if mean == 0 or std == 0: raise ValueError('Prior mean and std must be positive!')
            self.shape = mean**2/std**2
            self.scale = mean/std**2
            self.r , self.v = self.gamma_change(data = self.data, shape= self.shape, scale = self.scale)
        
        # Posterior Distribution
        self.interval = np.linspace(gamma.ppf(0.01,self.r, scale=1/self.v), gamma.ppf(0.999, self.r,scale=1/self.v),100)
        self.distribution = gamma.pdf(self.interval, a = self.r, scale = 1/self.v)
        
        # Data Summary
        self.mean, self.var = gamma.stats(self.r, scale = 1/self.v, moments='mv')
        self.median = gamma.median(self.r, scale = 1/self.v)
        self.mode = (self.r-1)/self.v
        self.up = gamma.ppf(0.75, self.r, scale = 1/self.v)
        self.low = gamma.ppf(0.25, self.r, scale = 1/self.v)
        self.IQR = self.up - self.low
        
        #Table
        self.summary = pd.Series({'Mean': self.mean,'Median': self.median,'Mode': self.mode,'Variance': self.var, 'IQR' : self.IQR})
        self.df = pd.DataFrame([self.summary])
    
    '''Print the Data Table'''
    def data_summarry(self):
        print(self.df)
    
    '''Take the probability of an interval'''
    def probability_calculation(self):
        print('Choose the type of probability')
        print(' Lower Probability - 1\n Upper Probability - 2 \n Interval Probability - 3')
        print('Insert the number of the option:')
        option = input()
        if option == '1':
            print('Insert mu value:')
            value = input()
            x = float(value)
            prob = gamma.cdf(x, self.r, scale = 1/self.v)
            print(f'The probability of mu < {x} is {prob}')
        elif option == '2':
            print('Insert mu value:')
            value = input()
            x = float(value)
            prob = 1 - gamma.cdf(x, self.r, scale = 1/self.v)
            print(f'The probability of mu > {x} is {prob}')
        elif option == '3':
            print('Insert lower value:')
            lower = input()
            print('Insert upper value:')
            upper = input()
            low = float(lower)
            up = float(upper)
            prob = gamma.cdf(up, self.r, scale = 1/self.v) - gamma.cdf(low, self.r, scale = 1/self.v)
            print(f'The probability of {low} < mu < {up} is {prob}')
        else:
            print("Invalid Input")
            sys.exit(0)
            