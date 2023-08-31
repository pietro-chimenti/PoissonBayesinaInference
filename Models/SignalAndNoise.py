#!/usr/bin/env python
"""This files provides basic tools to make bayesian inferences for poisson proceses with background only data and background plus signal.
"""


import math 
import numpy as np
from scipy.stats import gamma 
import emcee 
import arviz as az

class SignalAndNoise :
    """This class represent signal(on) and noise(off) parameter poissonian inference model for 
    bayesian analysis using Monte Carlo Markov Chain"""
    
    
    '''PRIOR both OFF and ON:'''  #é generico pois vamos depois aplicar para OFF e ON separados
        
    def log_prior_uniform(self, mu, alpha=0, beta=0): 
        self.log_mu = []
        for i in mu:
            if i > 0:
                self.log_mu.append(0)
            else:
                self.log_mu.append(- math.inf)   #retorno para ser rejeitado na função de aceptancia
        self.log_mu = np.array(self.log_mu)
        return self.log_mu        #retorna um array, mas selecionamos a entrada certa depois
    
    def log_prior_jeffrey(self, mu, alpha=0, beta=0):        
        self.log_mu = []
        for i in mu:
            if i > 0:
                self.log_mu.append(0.5*np.log(i))
            else:
                self.log_mu.append(- math.inf)
        self.log_mu = np.array(self.log_mu)
        return self.log_mu
        
    def log_prior_gamma(self,mu, alpha, beta):
        self.log_mu = []
        for i in mu:
            if i > 0:
                self.gamma = gamma.pdf(i, alpha, scale = 1/beta)
                self.log_mu.append(np.log(self.gamma))
            else:
                self.log_mu.append(- math.inf)
        self.log_mu = np.array(self.log_mu)
        return self.log_mu
        
    '''LIKELYHOOD'''  # estamos considerando uma entrada do tipo: mu = [mu_off, mu_on]
    
    def log_like_off(self,mu,data):
        self.data = np.array(data)
        self.factorial = []
        if mu[0] > 0:
            for value in self.data:
                if value <= 20 and value > 0 : #menor que vinte a função np.log consegue resolver numericamente
                    self.factorial.append(- np.log(math.factorial(value)))
                elif value > 20: #aprox de stirling 
                    self.factorial.append(-value*np.log(value)+value-0.5*np.log(2*value*math.pi))  
            self.factorial = np.array(self.factorial)
            return -len(self.data)*mu[0] + np.sum(self.data)*np.log(mu[0]) + np.sum(self.factorial)
        else:  
            return - math.inf
        
    def log_like_on(self,mu,data): 
        self.mu_off = mu[0]    
        self.mu_on = mu[1]
        self.data = np.array(data)
        self.factorial = []
        self.mu_sum = (self.mu_on + self.mu_off)
        
        if self.mu_off > 0 and self.mu_on > 0:
            for value in self.data:
                if value <= 20 and value > 0 : #menor que vinte a função np.log consegue resolver
                    self.factorial.append(- np.log(math.factorial(value)))
                elif value > 20:               #aprox de stirling
                    self.factorial.append(-value*np.log(value)+value-0.5*np.log(2*value*math.pi))  
            self.factorial = np.array(self.factorial)
            return -len(self.data)*self.mu_sum + np.sum(self.data)*np.log(self.mu_sum) + np.sum(self.factorial)
        else: 
            return - math.inf        
        
    def __init__(self, observed_value_off, observed_value_on, prior_off='uniform',
                 prior_on = 'uniform', mean_off=1, mean_on=1, std_off = 1, std_on=1):
        
        #constantes
        self.ndim, self.nwalkers = 2, 100
        
        #guarda o array de dados
        self.ov_off = np.array(observed_value_off)
        self.ov_on = np.array(observed_value_on)
        
        #seleciona o tipo de prior 
        if prior_off == 'uniform':
            self.log_prior_off = self.log_prior_uniform
        elif prior_off == 'jeffrey':
            self.log_prior_off = self.log_prior_jeffrey
        elif prior_off == 'gamma': 
            self.log_prior_off = self.log_prior_gamma
        else:
            print('Put a valid prior')
            
        if prior_on == 'uniform':
            self.log_prior_on = self.log_prior_uniform
        elif prior_on == 'jeffrey':
            self.log_prior_on = self.log_prior_jeffrey
        elif prior_on == 'gamma':
            self.log_prior_on = self.log_prior_gamma
        else:
            print('Put a valid prior')  
            
        #calcula o valor dos parametros da prior gamma 
        # CuiDADO: alpha ON and OFF
        self.alpha_on = mean_on**2/std_on**2
        self.beta_on = mean_on/std_on**2
        self.alpha_off = mean_off**2/std_off**2
        self.beta_off = mean_off/std_off**2
        
        # stats of a gamma sampler for p0
        m_off = np.mean(self.ov_off)
        dp_off = np.std(self.ov_off)
        m_on = np.mean(self.ov_on)
        dp_on = np.std(self.ov_on)

        #calcula o valor inicial a partir de uma amostragem gamma de parametros baseados nos dados
        self.p0 = np.array([np.random.gamma([m_off**2/dp_off**2,m_on**2/dp_on**2 ], #first value
                                                    scale=[dp_off**2/m_off,dp_on**2/m_on])])
                            
        for i in np.arange(self.nwalkers-1):
            gamma = np.random.gamma([m_off**2/dp_off**2,m_on**2/dp_on**2 ], 
                                                        scale=[dp_off**2/m_off,dp_on**2/m_on])
            self.p0 = np.append(self.p0,[gamma],axis=0)
        

    def log_posterior(self, mu):
        self.lp_on = self.log_prior_on(mu = mu,alpha = self.alpha_on, beta = self.beta_on)
        self.lp_off = self.log_prior_off(mu = mu,alpha = self.alpha_off, beta = self.beta_off)
        self.ll_on = self.log_like_off(mu = mu,data = self.ov_off)
        self.ll_off = self.log_like_on(mu = mu,data = self.ov_on)
            
        return float(self.lp_on[1]) + float(self.lp_off[0]) + self.ll_on + self.ll_off 
           
    def run (self, samples = 10000, burn_in = 1000):
        self.samples_list = []
        
        for i in range(5):
            print("Running chain n.",i)
            np.random.seed(7*i+1) #nenhuma razão, poderia ser qualquer numero
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior)
            
            self.state = sampler.run_mcmc(self.p0, burn_in, progress=(True)) #burn in 
            sampler.reset()
            
            sampler.run_mcmc(self.state, samples,progress=(True)) #real chain
            self.samples_list.append(sampler.get_chain(flat=True))
        
        return np.array(self.samples_list)
    
    
    
    
