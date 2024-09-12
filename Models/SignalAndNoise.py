#!/usr/bin/env python
"""This files provides basic tools to make bayesian inferences for poisson proceses with background only data and background plus signal.
"""
import math 
import numpy as np
import pandas as pd
from scipy.stats import gamma 
from scipy.stats import poisson
import matplotlib.pyplot as plt
import emcee 
import arviz as az
import xarray as xr
from scipy.special import gammaln
from scipy.optimize import minimize


class SignalAndNoise :
    """This class represent signal(on) and noise(off) parameter poissonian inference model for 
    bayesian analysis using Monte Carlo Markov Chain"""
    
    '''PRIOR OFF''' 
   
    def log_prior_uniform_off(self, mu, alpha=0, beta=0): #recebe uma lista mu = [mu_off,mu_on]
        self.mu_off = mu[0]    #seleciona a parte off do mu
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
            return np.log(gamma.pdf(self.mu_off, alpha, scale = 1/beta)) # log distribuição gamma
        else:
            return -math.inf
    
    '''PRIOR ON''' 
        
    def log_prior_uniform_on(self, mu, alpha=0, beta=0): 
        self.mu_on = mu[1]     #seleciona a parte on do mu
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
            return np.log(gamma.pdf(self.mu_on, alpha, scale = 1/beta))
        else:
            return - math.inf
    
    '''LIKELYHOOD'''  #estamos considerando uma entrada do tipo: mu = [mu_off, mu_on]
    
    def log_like_off(self,mu,data):
        self.data = np.array(data)   #transforma em array numpy caso não seja
        self.factorial = []          #cria lista que armazenara os valores do fatorial dos dados
        if mu[0] > 0:
            for value in self.data:
                poisson.logpmf
                if value <= 20 and value > 0 : #menor que vinte a função np.log consegue resolver numericamente
                    self.factorial.append(-np.log(math.factorial(value)))
                elif value > 20: #aproximação de stirling para log n!
                    self.factorial.append(-value*np.log(value)+value-0.5*np.log(2*value*math.pi))  
            self.factorial = np.array(self.factorial)  #transforma em array
            return -len(self.data)*mu[0] + np.sum(self.data)*np.log(mu[0]) + np.sum(self.factorial) #log da distribuição de poisson
        else:  
            return -math.inf
        
    def log_like_on(self,mu,data): 
        self.data = np.array(data)
        self.factorial = []
        self.mu_sum = (mu[1] + mu[0])
        
        if mu[0] > 0 and mu[1] > 0:
            for value in self.data:
                if value <= 20 and value > 0 : #menor que vinte a função np.log consegue resolver
                    self.factorial.append(- np.log(math.factorial(value)))
                elif value > 20:               #aprox de stirling
                    self.factorial.append(-value*np.log(value)+value-0.5*np.log(2*value*math.pi))  
            self.factorial = np.array(self.factorial)
            return -len(self.data)*self.mu_sum + np.sum(self.data)*np.log(self.mu_sum) + np.sum(self.factorial)
        else: 
            return - math.inf   
    
    '''PREDICTIVE'''
    
    def predictive_off(self, mu):
        if mu[0] > 0:
            return poisson.rvs(mu= mu[0],size = len(self.ov_off)) #gera valores aleatórios de poisson
        else:
            return np.full(len(self.ov_off),-2) #retorna um array de valor não importante, uma vez que será descartado na seleção da cadeia
        
    def predictive_on(self, mu):
        if mu[0] > 0 and mu[1] > 0:
            return poisson.rvs(mu= (mu[0] + mu[1]),size = len(self.ov_on)) #gera valores aleatórios de poisson
        else:
            return np.full(len(self.ov_off),-2)  #retorna um array de valor não importante, uma vez que será descartado na seleção da cadeia
        
    '''POSTERIOR'''
    
    def log_posterior(self, mu):
        
        """log prob"""
        self.lp_on = self.log_prior_on(mu = mu,alpha = self.alpha_on, beta = self.beta_on)
        self.lp_off = self.log_prior_off(mu = mu,alpha = self.alpha_off, beta = self.beta_off)
        self.ll_on = self.log_like_off(mu = mu,data = self.ov_off)
        self.ll_off = self.log_like_on(mu = mu,data = self.ov_on)
        
        self.log_post = self.lp_on + self.lp_off + self.ll_on + self.ll_off
        
        """blobs"""
        # Amostra Preditiva
        self.fake_dist_off = self.predictive_off(mu = mu)
        self.fake_dist_on = self.predictive_on(mu = mu)
        
        return self.log_post, self.fake_dist_off, self.fake_dist_on 
    
    '''CONSTRUCTOR'''
    
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
        
    '''RUN CHAINS'''
            
    def run (self, samples = 1000, burn_in = 500, n_chains = 8, nwalkers = 100):
        self.samples_list = []
        self.samples_list_flatten = []
        self.blobs_list = []
       
        # stats of a gamma sampler for p0
        m_off, dp_off = np.mean(self.ov_off), np.std(self.ov_off)
        m_on, dp_on = np.mean(self.ov_on), np.std(self.ov_on)
        self.p0 = np.array([np.random.gamma([m_off**2/dp_off**2,m_on**2/dp_on**2 ], #calcula o valor inicial a partir de uma amostragem gamma de parametros baseados nos dados
                                                    scale=[dp_off**2/m_off,dp_on**2/m_on])])
        for i in np.arange(nwalkers-1):
            gamma = np.random.gamma([m_off**2/dp_off**2,m_on**2/dp_on**2 ], 
                                                        scale=[dp_off**2/m_off,dp_on**2/m_on])
            self.p0 = np.append(self.p0,[gamma],axis=0)
            
        #run the chains
        for i in range(n_chains):
            print("Running chain n.",i+1)
            np.random.seed(42 + 7*i) #nenhuma razão, poderia ser qualquer o numero 
            self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_posterior)
            print("Burn_in progress:")
            self.state = self.sampler.run_mcmc(self.p0, burn_in, progress=(True)) #burn in 
            self.sampler.reset()
            print("Chain progress:")    
            self.sampler.run_mcmc(self.state, samples,progress=(True)) #real chain
            self.samples_list.append(self.sampler.get_chain(flat=False))
            self.samples_list_flatten.append(self.sampler.get_chain(flat=True))
            
            self.blobs_list.append(self.sampler.get_blobs(flat=True))
            
            
        self.n_chains = n_chains
        self.nwalkers = nwalkers
        self.samples = samples
        self.chains = np.array(self.samples_list)
        self.chains_flatten = np.array(self.samples_list_flatten)
        self.blobs_list = np.array(self.blobs_list)
        
        return self.chains
    
    '''ARVIZ DATA'''
    
    def full_arviz_dataset(self,labels=[r'$\mu_{off}$',r'$\mu_{on}$']):
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
        dataset = az.InferenceData(posterior = xrdata)
        return dataset
        
    def diff_seed_arviz_dataset(self,labels=[r'$\mu_{off}$',r'$\mu_{on}$']):
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
        dataset = az.InferenceData(posterior = xrdata)
        return dataset
    
    def single_chain_arvis_dataset(self,labels=[r'$\mu_{off}$',r'$\mu_{on}$']):
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
        dataset = az.InferenceData(posterior = xrdata)
        return dataset
    
    def full_dataset_flatten(self,labels=[r'$\mu_{off}$',r'$\mu_{on}$']):
        flatten = self.chains_flatten
        mu_off = flatten[:,:,0].flatten()
        mu_on = flatten[:,:,1].flatten()
        xrdata = xr.Dataset(
            data_vars = {
                  labels[0]: (["draw"],mu_off),
                  labels[1]: (["draw"],mu_on)
                 }
            )
        dataset = az.InferenceData(posterior = xrdata)
        
        return dataset
    
    '''SUMARIES'''
        
    def posterior_plot(self,ds):
        
        fig, ax = plt.subplots(1, 2)
        fig.suptitle("Distribuição a Posteriori",fontsize=17)
        
        az.plot_posterior(ds,var_names=[r'$\mu_{off}$'],ax=ax[0])
        az.plot_posterior(ds,var_names=[r'$\mu_{on}$'],ax=ax[1])
        
    
    def posterior_summary(self,ds):
        df = az.summary(ds,kind='stats')
        df.insert(0, 'param', [r'$\mu_{off}$', r'$\mu_{on}$'])
        df = df.round(2)
        print(df)
        fig, ax = plt.subplots(figsize=(9, 2))
        ax.axis('tight')
        ax.axis('off')
        tabela = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=['#f2f2f2']*len(df.columns))
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(12)
        tabela.scale(1.5, 1.5)
        plt.title("Medidas Resumo da Distribuição a Posteriori")
        plt.show()
        
        
    '''DIAGNOSE'''
    
    def diagnose(self, ds, single_chain,bins=20, title= ""):
        
        az.plot_trace(single_chain)
        plt.show()
        
        fig, ax = plt.subplots(1, 2)
        fig.suptitle("Autocorrelação",fontsize=20)
        
        az.plot_autocorr(single_chain,max_lag=200, combined=True,
                         var_names=[r'$\mu_{off}$'],ax=ax[0])
        az.plot_autocorr(single_chain,max_lag=200, combined=True,
                         var_names=[r'$\mu_{on}$'],ax=ax[1])
        plt.show()
        
        ess = az.ess(ds)
        r_hat = az.rhat(ds)

        summary = {
            'Stat': ['ess off','ess on','rhat off','rhat on'],
            'Mean': [float(ess[r'$\mu_{off}$'].mean()),float(ess[r'$\mu_{on}$'].mean()),float(r_hat[r'$\mu_{off}$'].mean()),float(r_hat[r'$\mu_{on}$'].mean())],
            'Std': [float(ess[r'$\mu_{off}$'].std()),float(ess[r'$\mu_{on}$'].std()),float(r_hat[r'$\mu_{off}$'].std()),float(r_hat[r'$\mu_{on}$'].std())]}
        
        df = pd.DataFrame(summary).round(3)
        print(df)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('tight')
        ax.axis('off')
        tabela = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=['#f2f2f2']*len(df.columns))
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(12)
        tabela.scale(1.5, 1.5)
        plt.title("Valores Diagnosticos das Cadeias")
        plt.show()


        fig, axs = plt.subplots(nrows=2, ncols=2)
        fig.suptitle("Distribuições Valores Diagnosticos das Cadeias",fontsize=20)

        axs[0, 0].hist(ess[r'$\mu_{off}$'], bins = bins, histtype='bar', facecolor='g',
                       alpha=0.7, edgecolor='black')
        axs[0, 0].set_title(r'Effect Sample Size $\mu_{off}$')
        axs[0, 0].set_xlabel("ess")
        axs[0, 0].set_ylabel("frequência")

        axs[1, 0].hist(ess[r'$\mu_{on}$'], bins = bins, histtype='bar', facecolor='g',
                       alpha=0.7, edgecolor='black')
        axs[1, 0].set_title(r'Effect Sample Size $\mu_{on}$')
        axs[1, 0].set_xlabel("ess")
        axs[1, 0].set_ylabel("frequência")

        axs[0, 1].hist(r_hat[r'$\mu_{off}$'], bins=bins, histtype='bar', facecolor='b',
                       alpha=0.7, edgecolor='black')
        axs[0, 1].set_title(r'$\^{R}$ $\mu_{off}$')
        axs[0, 1].set_xlabel(r"$\^{R}$")
        axs[0, 1].set_ylabel("frequência")

        axs[1, 1].hist(r_hat[r'$\mu_{on}$'], bins=bins, histtype='bar', facecolor='b',
                       alpha=0.7, edgecolor='black')
        axs[1, 1].set_title(r'$\^{R}$ $\mu_{on}$')
        axs[1, 1].set_xlabel(r"$\^{R}$")
        axs[1, 1].set_ylabel("frequência")
        
        plt.show()
        
    '''STATISTICAL ERROR'''    
    
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
    
    '''PREDICTIVE'''
    
    def preditive_distribution(self):
        
        self.flat_preditive_off = self.blobs_list[:,:,0,:].flatten()
        self.flat_preditive_on = self.blobs_list[:,:,1,:].flatten()
        
        mult = 10
        bins = 50
        xticks = 5
        rot = 45
        limite_off = [np.min(self.ov_off),np.max(self.ov_off)]
        limite_on = [np.min(self.ov_on),np.max(self.ov_on)]
        '''
        limite_off = [np.mean(self.flat_preditive_off) - mult*np.std(self.flat_preditive_off) ,
                      np.mean(self.flat_preditive_off) + mult*np.std(self.flat_preditive_off) ]
        
        limite_on = [np.mean(self.flat_preditive_on) - mult*np.std(self.flat_preditive_on) ,
                      np.mean(self.flat_preditive_on) + mult*np.std(self.flat_preditive_on) ]
        '''
        fig, ax = plt.subplots(1, 2)
        fig.suptitle("Amostragem Preditiva a Posteriori",fontsize=17)
        
        ax[0].set_title("Noise")
        ax[0].hist(self.flat_preditive_off,bins = np.linspace(np.min(self.flat_preditive_off),np.max(self.flat_preditive_off),bins), 
                   density=True,color='tomato', edgecolor='black', alpha=0.7, histtype='stepfilled',label='preditiva')
        ax[0].hist(self.ov_off, color="k", histtype="stepfilled", density = True, label="dados",alpha=0.5)
        ax[0].set_xticks(np.linspace(limite_off[0],  limite_off[1], xticks))
        ax[0].set_xlim(limite_off[0], limite_off[1])
        ax[0].tick_params(axis='x', which='major', labelsize=14, rotation=45)
        ax[0].set_xlabel("eventos")
        ax[0].set_ylabel("p.d.f.")
        
        ax[1].set_title("Signal")
        ax[1].hist(self.flat_preditive_on, bins = np.linspace(np.min(self.flat_preditive_on),np.max(self.flat_preditive_on),bins), 
                   density=True,color='skyblue', edgecolor='black', alpha=0.7, histtype='stepfilled',label='preditiva')
        ax[1].hist(self.ov_on, color="k", histtype="stepfilled", density = True, label="dados",alpha=0.5)
        ax[1].set_xticks(np.linspace(limite_on[0],  limite_on[1], xticks))
        ax[1].set_xlim(limite_on[0], limite_on[1])
        ax[1].tick_params(axis='x', which='major', labelsize=14, rotation=45)
        ax[1].set_xlabel("eventos")
        ax[1].set_ylabel("p.d.f.")
        
        ax[0].legend(loc='upper left',frameon=True,borderaxespad=0.)
        ax[1].legend(loc='upper left',frameon=True)
        plt.show()
        
        return self.flat_preditive_off, self.flat_preditive_on
        
    def aic(self):
        np.random.seed(42)
        data_off = self.ov_off
        data_on = self.ov_on
        x0 = [np.mean(self.ov_off),np.mean(self.ov_on)]
        
        # Função de log-verossimilhança para Poisson
        def log_likelihood_array(mu, data_off, data_on):
            self.ll_on = self.log_like_off(mu = mu,data = data_off)
            self.ll_off = self.log_like_on(mu = mu,data = data_on)
            return self.ll_on + self.ll_off
        
        # Estimativa de Máxima Verossimilhança
        result = minimize(log_likelihood_array,x0,args = (data_off,data_on))
        mu_estimado = [result.x[0],result.x[1]]
        
        def poisson_log_likelihood(lmbda, x):
            log_likelihood = -lmbda + x * np.log(lmbda) - math.lgamma(x + 1)
            return log_likelihood
        
        prob_list = []
        for i in data_off:
            i = np.array([i])
            prob = self.log_like_off(mu_estimado, i)
            prob_list.append(prob)
        for i in data_on:
            i = np.array([i])
            prob = self.log_like_on(mu_estimado, i)
            prob_list.append(prob)
            
        aic = -2*(sum(prob_list) - self.ndim)
        return aic

        
