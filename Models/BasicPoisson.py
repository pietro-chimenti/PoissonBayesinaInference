#!/usr/bin/env python
"""This files provides basic tools to make bayesian inferences on simple poissonian models.
"""

import sys
import numpy as np
from scipy import optimize
from scipy.stats import poisson
from scipy.stats import gamma 
import pandas as pd


class BasicPoisson :
    """This class represent a simple poissonian model for bayesian analysis"""

    def gamma_change(self,data, shape, scale):
        '''
        This function caculates the updated parameters analytically of a gamma
        posterior distribution Gamma(r,v).
            Parameters:
                data (int): observed value, used as likelyhood
                shape (int): first prior parameter
                scale (int): second prior parameter
            Returns:
                r (int): updater shape parameter
                v (int): updated scale parameter
        '''
        r = data + shape
        v = 1 + scale
        return r, v
    
    def __init__(self, observed_value, prior = 'uniform', mean = 0, std = 0, *args):
        '''
        Receive the users data and the prior used (Uniform, Jeffrey´s, Gamma), 
        and calculate the posterior distribution.
        '''
    
        # Users Data
        self.ov = observed_value
        
        # New Parameters
        if prior == 'uniform':
            self.r , self.v = self.gamma_change(data = self.ov, shape= 0.5, scale = 0)
        if prior == 'jeffreys':
            self.r , self.v = self.gamma_change(data = self.ov, shape= 1, scale = 0)
        if prior == 'gamma':
            if mean == 0 or std == 0: raise ValueError('Prior mean and std must be positive!')
            self.shape = mean**2/std**2
            self.scale = mean/std**2
            self.r , self.v = self.gamma_change(data = self.ov, shape= self.shape, scale = self.scale)
        
        # Posterior Distribution
        self.interval = np.linspace(gamma.ppf(0.0001,self.r, scale=1/self.v), gamma.ppf(0.9999, self.r,scale=1/self.v),100)
        self.distribution = gamma.pdf(self.interval, a = self.r, scale = 1/self.v)
        
        # Data Summary
        self.mean, self.var = gamma.stats(self.r, scale = 1/self.v, moments='mv')
        self.median = gamma.median(self.r, scale = 1/self.v)
        self.mode = (self.r-1)/self.v
        self.up = gamma.ppf(0.75, self.r, scale = 1/self.v)
        self.low = gamma.ppf(0.25, self.r, scale = 1/self.v)
        self.IQR = self.up - self.low
        
        # Summary Table
        self.summary = pd.Series({'Mean': self.mean,'Median': self.median,'Mode': self.mode,'Variance': self.var,'Low Quartile':self.low,'Up Quartile': self.up ,'IQR' : self.IQR})
        self.df = pd.DataFrame([self.summary])
    
    def data_summarry(self):
        """ Prints the posterior distribution summaries"""
        
        print("Posterior Summaries")
        print(self.df.T)
        
    def credible_interval(self,trust = 0.95, option = 1):
        '''
        Calculate the Credible Interval of the Posterior Gamma Distribution.
            Parameters:
                trust(float): confidence interval choosen by the user
                option(int): 1- Upper Limit Interval; 2- Symmetrical Interval; 
                3- Highest Density Interval
            Returns:
                lower_limit(float): lower point of the interval
                upper_limit(float): highest point of the interval
        '''
        if option == 1:
            lower_limit = gamma.ppf(0.0001, self.r, scale = 1/self.v)
            upper_limit = gamma.ppf(trust, self.r, scale = 1/self.v)
            print(f'The upper limit credible interval is: ({lower_limit},{upper_limit})')
            return lower_limit, upper_limit
        elif option == 2:
            itv = (1 - trust)/2
            lower_limit = gamma.ppf(itv, self.r, scale = 1/self.v)
            upper_limit = gamma.ppf(1 - itv , self.r, scale = 1/self.v)
            print(f'The Symmetrical credible interval is: ({lower_limit},{upper_limit})')
            return lower_limit, upper_limit
        elif option == 3:
            
            def interval_width(lower):
                '''
                For lower limit, calculate the variation between the upper and lower
                separeted by the credible interval.
                '''
                upper = gamma.ppf(gamma.cdf(lower, self.r, scale = 1/self.v) + trust, self.r, scale = 1/self.v)
                return upper - lower
            #média simetrica como chute
            initial_guess = gamma.ppf((1-trust)/2, self.r, scale = 1/self.v)
            #acha os valores que minimiza a função 
            optimize_result = optimize.minimize(interval_width, initial_guess)
            #retira o valor correspondente
            lower_limit = optimize_result.x[0]
            width = optimize_result.fun
            #acha o valor superior somando ao intervalo
            upper_limit = lower_limit + width
            print(f'The High Density Credible Interval is: ({lower_limit},{upper_limit})')
            return lower_limit, upper_limit
        else:
            print("Invalid Input")
            sys.exit(0)
