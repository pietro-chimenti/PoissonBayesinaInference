#!/usr/bin/env python
"""This files provides basic tools to make bayesian inferences for poisson proceses with background only data and background plus signal.
"""

import math 
import numpy as np
from scipy.stats import gamma 
import emcee 
import arviz as az
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import poisson

class SignalAndNoise :
    """This class represent signal(on) and noise(off) parameter poissonian inference model for 
    bayesian analysis using Monte Carlo Markov Chain"""
    
    
    '''PRIOR both OFF and ON:'''  #é generico pois vamos depois aplicar para OFF e ON separados
        
    def log_prior_uniform_off(self, mu, alpha=0, beta=0): 
        self.mu_off = mu[0]
        if self.mu_off > 0:
            return 0
        else:
            return -math.inf   #retorno para ser rejeitado na função de aceptancia
    
    def log_prior_jeffrey_off(self, mu, alpha=0, beta=0):        
        self.mu_off = mu[0]
        if self.mu_off > 0:
            return 0.5*np.log(self.mu_off)
        else:
            return -math.inf
        
    def log_prior_gamma_off(self, mu, alpha, beta):
        self.mu_off = mu[0]
        if self.mu_off > 0:
            self.gamma = gamma.pdf(self.mu_off, alpha, scale = 1/beta)
            return np.log(self.gamma)
        else:
            return - math.inf
        
    def log_prior_uniform_on(self, mu, alpha=0, beta=0): 
        self.mu_on = mu[1]
        if self.mu_on > 0:
            return 0
        else:
            return -math.inf   
    
    def log_prior_jeffrey_on(self, mu, alpha=0, beta=0):        
        self.mu_on = mu[1]
        if self.mu_on > 0:
            return 0.5*np.log(self.mu_on)
        else:
            return -math.inf
        
    def log_prior_gamma_on(self, mu, alpha, beta):
        self.mu_on = mu[1]
        if self.mu_on > 0:
            self.gamma = gamma.pdf(self.mu_on, alpha, scale = 1/beta)
            return np.log(self.gamma)
        else:
            return - math.inf
    
        
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
    
    '''predictive'''
    
    def predictive_off(self, mu):
        if mu[0] > 0:
            return poisson.rvs(mu= mu[0],size = 1)
        else:
            return -1
        
    def predictive_on(self, mu):
        if mu[1] > 0:
            return poisson.rvs(mu= mu[1],size = 1)
        else:
            return -1
        
    def __init__(self, observed_value_off, observed_value_on, prior_off='uniform',
                 prior_on = 'uniform', mean_off=1, mean_on=1, std_off = 1, std_on=1):
        #constantes
        self.ndim =  2
        
        #guarda o array de dados
        self.ov_off = np.array(observed_value_off)
        self.ov_on = np.array(observed_value_on)
        
        #seleciona o tipo de prior 
        if prior_off == 'uniform':
            self.log_prior_off = self.log_prior_uniform_off
        elif prior_off == 'jeffrey':
            self.log_prior_off = self.log_prior_jeffrey_off
        elif prior_off == 'gamma': 
            self.log_prior_off = self.log_prior_gamma_off
        else:
            print('Put a valid prior')
            
        if prior_on == 'uniform':
            self.log_prior_on = self.log_prior_uniform_on
        elif prior_on == 'jeffrey':
            self.log_prior_on = self.log_prior_jeffrey_on
        elif prior_on == 'gamma':
            self.log_prior_on = self.log_prior_gamma_on
        else:
            print('Put a valid prior')  
            
        #calcula o valor dos parametros da prior gamma 
        self.alpha_on = mean_on**2/std_on**2
        self.beta_on = mean_on/std_on**2
        self.alpha_off = mean_off**2/std_off**2
        self.beta_off = mean_off/std_off**2
        
    def log_posterior(self, mu):
        
        """log prob"""
        self.lp_on = self.log_prior_on(mu = mu,alpha = self.alpha_on, beta = self.beta_on)
        self.lp_off = self.log_prior_off(mu = mu,alpha = self.alpha_off, beta = self.beta_off)
        self.ll_on = self.log_like_off(mu = mu,data = self.ov_off)
        self.ll_off = self.log_like_on(mu = mu,data = self.ov_on)
        
        self.log_post = self.lp_on + self.lp_off + self.ll_on + self.ll_off
        
        """blobs"""
    
        self.fake_dist_off = self.predictive_off(mu = mu)
        self.fake_dist_on = self.predictive_on(mu = mu)
        
        return self.log_post, self.fake_dist_off, self.fake_dist_on
            
    def run (self, samples = 10000, burn_in = 1000, n_chains = 8, nwalkers = 100 ):
        self.samples_list = []
        self.samples_list_flatten = []
       
        
        # stats of a gamma sampler for p0
        m_off, dp_off = np.mean(self.ov_off), np.std(self.ov_off)
        m_on, dp_on = np.mean(self.ov_on), np.std(self.ov_on)
        #calcula o valor inicial a partir de uma amostragem gamma de parametros baseados nos dados
        self.p0 = np.array([np.random.gamma([m_off**2/dp_off**2,m_on**2/dp_on**2 ], #first value
                                                    scale=[dp_off**2/m_off,dp_on**2/m_on])])
        
        for i in np.arange(nwalkers-1):
            gamma = np.random.gamma([m_off**2/dp_off**2,m_on**2/dp_on**2 ], 
                                                        scale=[dp_off**2/m_off,dp_on**2/m_on])
            self.p0 = np.append(self.p0,[gamma],axis=0)
        #run the chains
        for i in range(n_chains):
            print("Running chain n.",i)
            np.random.seed(42 + 7*i) #nenhuma razão, poderia ser qualquer o numero 
            
            dtype = [("pred_off", int), ("pred_on", int)]
            
            self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_posterior, blobs_dtype=dtype)
            print("Burn_in progress:")
            self.state = self.sampler.run_mcmc(self.p0, burn_in, progress=(True)) #burn in 
            self.sampler.reset()
            print("Chain progress:")    
            self.sampler.run_mcmc(self.state, samples,progress=(True)) #real chain
            self.samples_list.append(self.sampler.get_chain(flat=False))
            self.samples_list_flatten.append(self.sampler.get_chain(flat=True))
            
        self.n_chains = n_chains
        self.nwalkers = nwalkers
        self.samples = samples
        self.chains = np.array(self.samples_list)
        self.chains_flatten = np.array(self.samples_list_flatten)
        
        return self.chains
    
    def full_arviz_dataset(self,labels=["mu_off","mu_on"]):
        mu_off = self.chains[:,:,:,0]
        mu_on = self.chains[:,:,:,1]
        xrdata = xr.Dataset(
            data_vars = {
                  labels[0]: (["chain","draw","walker"],mu_off),
                  labels[1]: (["chain","draw","walker"],mu_on)
                 },
            coords= {
                "chain": (["chain"],np.arange(self.n_chains)),
                "draw":(["draw"],np.arange(self.samples)),
                "walker": (["walker"],np.arange(self.nwalkers))
                }
            )
        '''
        xrobs = xr.Dataset(
            data_vars = {
                  labels[0]: (["data"],self.ov_off),
                  labels[1]: (["data"],self.ov_on)
                 }
            )
        '''
        dataset = az.InferenceData(posterior = xrdata)
        return dataset
        
    def diff_seed_arviz_dataset(self,labels=["mu_off","mu_on"]):
        mu_off = self.chains[:,:,0,0]
        mu_on = self.chains[:,:,0,1]
        xrdata = xr.Dataset(
            data_vars = {
                  labels[0]: (["chain","draw"],mu_off),
                  labels[1]: (["chain","draw"],mu_on)
                 },
            coords= {
                "chain": (["chain"],np.arange(self.n_chains)),
                "draw":(["draw"],np.arange(self.samples)),
                }
            )
        '''
        xrobs = xr.Dataset(
            data_vars = {
                  labels[0]: (["data"],self.ov_off),
                  labels[1]: (["data"],self.ov_on)
                 }
            )
        '''
        dataset = az.InferenceData(posterior = xrdata)
        return dataset
    
    def single_chain_arvis_dataset(self,labels=["mu_off","mu_on"]):
        ch = self.chains
        self.tr_chains = ch.transpose(0,2,1,3)
        mu_off = self.tr_chains[0,:,:,0]
        mu_on = self.tr_chains[0,:,:,1]
        xrdata = xr.Dataset(
            data_vars = {
                  labels[0]: (["chain","draw"],mu_off),
                  labels[1]: (["chain","draw"],mu_on)
                 },
            coords= {
                "chain": (["chain"],np.arange(self.nwalkers)),
                "draw":(["draw"],np.arange(self.samples)),
                }
            )
        '''
        xrobs = xr.Dataset(
            data_vars = {
                  labels[0]: (["data"],self.ov_off),
                  labels[1]: (["data"],self.ov_on)
                 }
            )
        '''
        dataset = az.InferenceData(posterior = xrdata)
        return dataset
    
    def diagnose(self, ds, single_chain,bins=20, title= ""):
        
        az.plot_trace(single_chain)
        plt.show()
        
        az.plot_autocorr(single_chain,max_lag=200, combined=True)
        plt.show()
        
        ess = az.ess(ds)
        r_hat = az.rhat(ds)

        summary = {
            'Stat': ['ess off','ess on','rhat off','rhat on'],
            'Mean': [float(ess['mu_off'].mean()),float(ess['mu_on'].mean()),float(r_hat['mu_off'].mean()),float(r_hat['mu_on'].mean())],
            'Std': [float(ess['mu_off'].std()),float(ess['mu_on'].std()),float(r_hat['mu_off'].std()),float(r_hat['mu_on'].std())]
        }

        df = pd.DataFrame(summary).round(5)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('tight')
        ax.axis('off')
        tabela = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=['#f2f2f2']*len(df.columns))
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(12)
        tabela.scale(1.5, 1.5)
        plt.title(title)
        plt.show()

        fig, axs = plt.subplots(nrows=2, ncols=2)

        axs[0, 0].hist(ess['mu_off'], bins = bins, histtype='stepfilled', facecolor='g',
                       alpha=0.75)
        axs[0, 0].set_title('effect sample size mu_off')
        axs[0, 0].set_xlabel("n eff sample size")
        axs[0, 0].set_ylabel("frequence")

        axs[1, 0].hist(ess['mu_on'], bins = bins, histtype='stepfilled', facecolor='g',
                       alpha=0.75)
        axs[1, 0].set_title('effect sample size mu_on')
        axs[1, 0].set_xlabel("n eff sample size")
        axs[1, 0].set_ylabel("frequence")

        axs[0, 1].hist(r_hat['mu_off'], bins=bins, histtype='stepfilled', facecolor='b',
                       alpha=0.75)
        axs[0, 1].set_title('r^ mu_off')
        axs[0, 1].set_xlabel("r^")
        axs[0, 1].set_ylabel("frequence")

        axs[1, 1].hist(r_hat['mu_on'], bins=bins, histtype='stepfilled', facecolor='b',
                       alpha=0.75)
        axs[1, 1].set_title('r^ mu_on')
        axs[1, 1].set_xlabel("r^")
        axs[1, 1].set_ylabel("frequence")
        
        plt.suptitle(title, fontsize=18)
        plt.show()
        
        
    def statistic_error(self):
        """desvio da média"""
        #média entre as cadeias de seed diferente 
        mu_off_mean = np.mean(self.chains_flatten[:,:,0], axis = 1)
        mu_on_mean = np.mean(self.chains_flatten[:,:,1], axis = 1)
        
        #cria um array com os valores repetidos da média
        mu_off_mean_total = np.full(len(mu_off_mean),np.mean(mu_off_mean),dtype=float)
        mu_on_mean_total = np.full(len(mu_on_mean),np.mean(mu_on_mean),dtype=float)

        # variancia da média
        error_mean_off = np.sum(np.square(mu_off_mean - mu_off_mean_total))/(self.n_chains-1)
        error_mean_on = np.sum( np.square(mu_on_mean - mu_on_mean_total))/(self.n_chains-1)
        
        #desvio padrão da média
        desv_mean_off = math.sqrt(error_mean_off)
        desv_mean_on = math.sqrt(error_mean_on)
        
        """desvio da variancia"""
        
        #variancia entre as cadeias de seed diferente 
        mu_off_var = np.var(self.chains_flatten[:,:,0], axis = 1)
        mu_on_var = np.var(self.chains_flatten[:,:,1], axis = 1)
        
        #cria um array com os valores repetidos da média
        mu_off_var_total = np.full(len(mu_off_var),np.mean(mu_off_var),dtype=float)
        mu_on_var_total = np.full(len(mu_on_var),np.mean(mu_on_var),dtype=float)

        # variancia da média
        error_var_off = np.sum(np.square(mu_off_var - mu_off_var_total))/(self.n_chains-1)
        error_var_on = np.sum( np.square(mu_on_var - mu_on_var_total))/(self.n_chains-1)
        
        #desvio padrão da média
        desv_var_off = math.sqrt(error_var_off)
        desv_var_on = math.sqrt(error_var_on)
        
        summary = {
            'Desv': ['desv mean off','desv mean on','desv var off','desv var off'],
            'Value': [desv_mean_off,desv_mean_on,desv_var_off,desv_var_off]
        }

        df_desv = pd.DataFrame(summary)
        print(df_desv)
        
        return desv_mean_off, desv_mean_on, desv_var_off, desv_var_on, df_desv
    
    
    def preditive_distribution(self):
        self.flat_blobs = self.sampler.get_blobs(flat=True)
        self.flat_preditive_off = self.flat_blobs["pred_off"]
        self.flat_preditive_on = self.flat_blobs["pred_on"]
        
        
        plt.hist(self.flat_preditive_off,bins = np.arange(0,15), density=True)
        plt.title("Mu_off posterior predictive distribuion")
        plt.show()
        
        plt.hist(self.flat_preditive_on, bins = np.arange(0,15), density=True)
        plt.title("Mu_on posterior predictive distribuion")
        plt.show()
        
        
        return self.flat_preditive_off, self.flat_preditive_on
        
