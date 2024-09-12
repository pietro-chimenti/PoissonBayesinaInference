# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:14:34 2024

@author: orion
"""

import math 
import numpy as np
import pandas as pd
from scipy.stats import gamma 
from scipy.stats import poisson
import matplotlib.pyplot as plt
import emcee
import corner 
import arviz as az
import h5py
from numba import jit 

class Signal_and_Hierachical_Noise_2:
    
    '''PRIOR''' 
    
    def log_prior_signal_uniform(self, param): 
        
        self.signal = param[0]   
        
        if self.signal > 0:
            return 0
        else:
            return -math.inf   
        
        
    def log_prior_signal_jeff(self, param):
        
        self.signal = param[0]
        
        if self.signal > 0:
            return -0.5*np.log(self.signal)
        else:
            return -math.inf
    
    def log_prior_mu_noise_uniform(self, param): 
        
        self.mu_noise = param[1]     
        
        if self.mu_noise > 0:
            return 0
        else:
            return -math.inf   
        
        
    def log_prior_sigma_noise_uniform(self, param): 
        
        self.sigma_noise = param[2]    
        
        if self.sigma_noise > 0:
            return 0
        else:
            return -math.inf   
        
    def log_prior_alpha_noise_uniform(self, param): 
        
        self.alpha_noise = param[3]    
        
        if self.alpha_noise > 0:
            return 0
        else:
            return -math.inf   
        
        
    '''POPULATION'''
    
    
    def log_population_noise(self,param):
        self.noise = param[4:]
        self.mu_noise = param[1]
        self.sigma_noise = param[2]
        noise_array = np.array(self.noise)
        log_gamma_list = np.array([])
    
        alpha = self.mu_noise**2/self.sigma_noise**2
        beta = self.mu_noise/self.sigma_noise**2
        
        if np.all(noise_array > 0) and self.mu_noise>0 and self.sigma_noise>0 :
            for i in self.noise:
                log_pdf = gamma.logpdf(i, alpha, scale = 1/beta)
                log_gamma_list = np.append(log_gamma_list, log_pdf)
            return np.sum(log_gamma_list)
        else:
            return -math.inf
    
    '''LIKELYHOOD'''
    
    
    def log_like_off(self, param, data_off):
        
        self.noise_off = param[4: 4 + len(data_off)]
        like_off_list = np.array([])
        noise_off_array = np.array(self.noise_off)
        
        if np.all(noise_off_array > 0):
            for i in data_off:
                mu = self.noise_off[data_off.index(i)] #aqui tem problema para dados repetidos
                log_poisson = poisson.logpmf(i,mu)
                like_off_list = np.append(like_off_list, log_poisson)
            return np.sum(like_off_list)
        else:
            return -math.inf
        
    
    def log_like_on(self, param, data_off, data_on):#depende de data off só para seleção dos parametros
            
        self.signal = param[0]
        self.noise_on = param[len(data_off)+4 : ]
        noise_on_array = np.array(self.noise_on)
        like_on_list = np.array([])
            
        if np.all(noise_on_array > 0) and self.signal > 0:
            for i in data_on:
                mu = self.noise_on[data_on.index(i)] + self.signal
                log_poisson = poisson.logpmf(i,mu)
                like_on_list = np.append(like_on_list, log_poisson)
            return np.sum(like_on_list)
        else:
            return -math.inf
        
    def log_like_noise(self, param, data_noise):
        self.alpha_noise = param[3]
        self.noise_total = param[4 : ]
        noise_total_array = np.array(self.noise_total)
        like_noise_list = []
        
        if np.all(noise_total_array > 0) and self.alpha_noise > 0:
            for i in data_noise:
                mu = self.noise_total[data_noise.index(i)]*self.alpha_noise #aqui tem problema para dados repetidos
                log_poisson = poisson.logpmf(i,mu)
                like_noise_list = np.append(like_noise_list, log_poisson)
            return np.sum(like_noise_list)
        else:
            return -math.inf
        
    '''POSTERIOR'''
    
   
    def log_posterior(self, param):
            
        """log prob"""
            
        self.lp_signal = self.log_prior_signal(param = param)
        self.lp_mu = self.log_prior_mu_noise_uniform(param = param)
        self.lp_sigma = self.log_prior_sigma_noise_uniform(param = param)
        self.lp_al = self.log_prior_alpha_noise_uniform(param = param)
        self.lpopulation = self.log_population_noise(param = param)
        self.ll_off = self.log_like_off(param = param,data_off = self.observed_value_off)
        self.ll_on = self.log_like_on(param = param,data_off = self.observed_value_off,data_on=self.observed_value_on)
        self.ll_noise = self.log_like_noise(param = param, data_noise = self.observed_value_noise)
        self.log_post = self.lp_signal + self.lp_mu + self.lp_sigma + self.lp_al + self.lpopulation + self.ll_on + self.ll_off + self.ll_noise
        
        return self.log_post
    
    '''CONSTRUCTOR'''
    
    def __init__(self,observed_value_off, observed_value_on, observed_value_noise, samples, nwalkers, prior_signal='uniform'):
        
        #salva dados da cadeia
        self.samples = samples
        self.nwalkers = nwalkers
        
        #dimensão do espaço de parametros
        self.ndim =  4 + len(observed_value_off) + len(observed_value_on) 
        self.n_dim_noise =  len(observed_value_off) + len(observed_value_on)
        
        
        #nome dos parametros:
        self.label_interest = [r'$S$', r'$\mu_{noise}$', r'$\sigma_{noise}$',r'$\alpha_{cut}$']
        self.label_noise = []
        for i in range(self.n_dim_noise):
            stg = f'$R_{{i}}$'
            self.label_noise.append(stg)
        self.label_total = self.label_interest + self.label_noise 
            
        #guarda os dados:
        self.observed_value_off = observed_value_off
        self.observed_value_on = observed_value_on
        self.observed_value_noise = observed_value_noise

        self.ov_off = np.array(observed_value_off)
        self.ov_on = np.array(observed_value_on)
        self.ov_noise = np.array(observed_value_noise)
            
        #seleciona o tipo de prior para signal
        if prior_signal == 'uniform':
            self.log_prior_signal = self.log_prior_signal_uniform
        elif prior_signal == 'jeffrey':
            self.log_prior_signal = self.log_prior_signal_jeff
        else:
             print('Put a valid prior for signal parameter')
             
             
    '''RUN CHAIN'''
    
    
    def run (self, save = False, filename=''):
        np.random.seed(42)
        self.matrix_p0 = np.empty((0,self.ndim))
        
        # stats of a gamma sampler for p0
        m_off, dp_off = np.mean(self.ov_off), np.std(self.ov_off)
        m_on, dp_on = np.mean(self.ov_on), np.std(self.ov_on)
        m_noise, dp_noise = np.mean(self.ov_noise), np.std(self.ov_noise)
        data_total = np.concatenate((self.ov_off,self.ov_on))
        
        
        for i in np.arange(self.nwalkers):
            p0_signal = np.random.gamma(m_on**2/dp_on**2,scale=dp_on**2/m_on)
            p0_mu_noise = np.random.gamma(m_off**2/dp_off**2,scale=dp_off**2/m_off)
            p0_sigma_noise = np.random.gamma(dp_off**2/m_off,scale=dp_off**2/m_off)
            p0_alpha_noise = np.random.gamma(m_noise**2/dp_noise**2,scale=dp_noise**2/m_noise)
            self.list_p0 = [p0_signal,p0_mu_noise,p0_sigma_noise,p0_alpha_noise]
            
            for i in range(self.n_dim_noise):
                noise = np.random.gamma(data_total[i]**2/dp_off**2,scale=dp_off**2/m_off) #dados on terão inicialização mais acima
                self.list_p0.append(noise)
                
            self.array_p0 = np.array([self.list_p0])
            self.matrix_p0 = np.append(self.matrix_p0,self.array_p0,axis=0)

        #run the chains 
        
        if save == True:      #save the chain
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(self.nwalkers, self.ndim)
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior,backend=backend)
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior)
        
        print("Chain progress:")
        self.sampler.run_mcmc(self.matrix_p0, self.samples, progress=(True))
        
        
    def read_saved_chain(self,filename):
        
        self.sampler = emcee.backends.HDFBackend(filename)
        
    def get_chain(self, burn_in):
        self.burn_in = burn_in
        self.chains = self.sampler.get_chain(discard=burn_in)
        self.chains_flat= self.sampler.get_chain(flat=True, discard=burn_in)
        self.chains_flat_no_discard = self.sampler.get_chain(flat=True)
        
        #arviz data
        dataset = az.from_emcee(self.sampler, var_names=self.label_total)
        self.dataset = dataset.sel(draw=slice(self.burn_in, None),inplacebool= True)
        
        
        #preditive
        mu_signal  = self.chains_flat[:, 0]
        mu_noise = self.chains_flat[:, 1]
        desv_noise = self.chains_flat[:, 2]
        alpha_cut = self.chains_flat[:,3]
        
        alpha = mu_noise**2/desv_noise**2
        beta = mu_noise/desv_noise**2
        noise_param_off = gamma.rvs(alpha, scale = 1/beta)
        
        
        self.pred_off = poisson.rvs(mu = noise_param_off)
        self.pred_on = poisson.rvs(mu = noise_param_off + mu_signal)
        self.pred_cut = poisson.rvs(mu = noise_param_off*alpha_cut)
        
        return self.chains_flat, self.chains, self.chains_flat_no_discard, self.dataset
    
    
    '''SIMULATION DIAGNOSE'''
    
    def acceptance_fraction(self):
        
        self.acceptance_fraction = self.sampler.acceptance_fraction

        print("Mean acceptance fraction: {0:.3f}".format(
        np.mean(self.acceptance_fraction)))

        return self.acceptance_fraction
    
    def autocorrelation_time(self):
        
        self.autocorrelation_time = self.sampler.get_autocorr_time(discard=self.burn_in,
                                                                   quiet=True)
        print("Mean autocorrelation time: {0:.3f} steps".format(
        np.mean(self.autocorrelation_time)))
 
        
        return self.autocorrelation_time
    
    '''BASIC GRAPHICS'''    
    
    def trace_plot(self, noise_axes=10):
        az.style.use(["arviz-darkgrid", "arviz-cyanish"])
        fig, axes = plt.subplots(4, figsize=(10, 6), sharex=True)
        for i in range(4):
            ax = axes[i]
            ax.plot(self.chains_flat_no_discard[:, i], 'k')
            ax.set_xlim(0, len(self.chains_flat_no_discard))
            ax.set_ylabel(self.label_interest[i], fontsize=16)
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ax.axvline(x = self.burn_in*self.nwalkers, color = 'r', alpha = 0.9, label = 'burn-in line')
            ax.legend()
        fig.suptitle('Trace of average interest parameters', fontsize=20)
        axes[-1].set_xlabel("Iterations", fontsize=17);

        
        label_noise = []
        
        for i in range(self.n_dim_noise):
            stg = f'$R_{{{i + 1}}}$'
            label_noise.append(stg)
            
         
        for i in range(0,self.n_dim_noise,noise_axes):
            fig, axs = plt.subplots(noise_axes, figsize=(10, 6), sharex=True)
            for j in range(noise_axes):
                if j+i< self.n_dim_noise:
                    ax = axs[j]
                    ax.plot(self.chains_flat_no_discard[:,i+j+4], "k")
                    ax.set_xlim(0, len(self.chains_flat_no_discard))
                    ax.set_ylabel(label_noise[i+j],fontsize=16)
                    ax.set_yticklabels([])
                    ax.yaxis.set_label_coords(-0.1, 0.5)
                    ax.axvline(x =self.burn_in*self.nwalkers, color = 'r', alpha = 0.9, label = 'burn-in line')
            
            fig.suptitle(r'Trace of average noise rate parameters $R$ ', fontsize=20)
            plt.legend(loc='upper right')
            axs[-1].set_xlabel("Iterations", fontsize=17);
            
            
    
    def posterior_graph_interest_params(self):
        
        plt.hist(self.chains_flat[:, 0], 100, color="k", histtype="step")
        plt.title(r" Distribuição a Posteriori $\mu_{signal}$")
        plt.gca().set_yticks([])
        plt.tight_layout()
        plt.show()

        plt.hist(self.chains_flat[:, 1], 100, color="k", histtype="step")
        plt.title(r" Distribuição a Posteriori $\mu_{noise}$")
        plt.gca().set_yticks([])
        plt.tight_layout()
        plt.show()

        plt.hist(self.chains_flat[:, 2], 100, color="k", histtype="step")
        plt.title(r"Distribuição a Posteriori $\sigma_{noise}$" )
        plt.gca().set_yticks([])
        plt.tight_layout()
        plt.show()
        
        plt.hist(self.chains_flat[:, 3], 100, color="k", histtype="step")
        plt.title(r"Distribuição a Posteriori $\alpha_{noise}$" )
        plt.gca().set_yticks([])
        plt.tight_layout()
        plt.show()
            
    def posterior_graph_noise_params(self,num_columns = 4):

        num_lines = int(np.ceil(self.n_dim_noise / num_columns))
        fig, axs = plt.subplots(num_lines, num_columns, figsize=(2.4*num_lines, 1.5*num_lines))  

        for i, ax in enumerate(axs.flat):
            if i <= (self.n_dim_noise-1):
                ax.hist(self.chains_flat[:, i + 4], bins=100, color="k", histtype="step")
                ax.set_title(f" Noise Parameter {i + 1}")

        fig.suptitle('Posterior sampling of the noise parameters', fontsize=20)
        plt.tight_layout()
        plt.show()
        
        
    def corner_interest_param(self):
        
        fig = corner.corner(self.chains_flat[:,:4], labels = self.label_interest)
        
        
    def pred_graph(self):
        
        font_ax = 13
        counts, bins = np.histogram(self.pred_off, bins=100)
        weights = counts / np.max(counts)
        plt.hist(bins[:-1], bins, weights=weights,color='tomato', edgecolor='black', alpha=0.7, histtype='stepfilled',label='predictive')
        
        counts, bins = np.histogram(self.observed_value_off, bins=10)
        weights = counts / np.max(counts)
        plt.hist(bins[:-1], bins, weights=weights,color="k", histtype="stepfilled", 
         label="data",alpha=0.5)

        plt.title(" Predictive sampling OFF")
        plt.xlabel('Events',fontsize=font_ax)
        plt.ylabel('Frequency',fontsize=font_ax)
        plt.legend()
        plt.show()

        counts, bins = np.histogram(self.pred_on, bins=100)
        weights = counts / np.max(counts)
        plt.hist(bins[:-1], bins, weights=weights,color='skyblue', edgecolor='black', alpha=0.7, histtype='stepfilled',label='predictive')
        
        counts, bins = np.histogram(self.observed_value_on, bins=20)
        weights = counts / np.max(counts)
        plt.hist(bins[:-1], bins, weights=weights,color="k", histtype="stepfilled", 
         label="data",alpha=0.5)
        
        plt.title("Predictive sampling ON")
        plt.xlabel('Events',fontsize=font_ax)
        plt.ylabel('Frequency',fontsize=font_ax)
        plt.legend()
        plt.show()
        
        counts, bins = np.histogram(self.pred_cut, bins=100)
        weights = counts / np.max(counts)
        plt.hist(bins[:-1], bins, weights=weights,color='green', edgecolor='black', alpha=0.7, histtype='stepfilled',label='predictive')
        
        counts, bins = np.histogram(self.observed_value_noise, bins=20)
        weights = counts / np.max(counts)
        plt.hist(bins[:-1], bins, weights=weights,color="k", histtype="stepfilled", 
         label="data",alpha=0.5)
        
        plt.title("Predictive sampling CUT")
        plt.xlabel('Events',fontsize=font_ax)
        plt.ylabel('Frequency',fontsize=font_ax)
        plt.legend()
        plt.show()
        
        
    def summary_sample_interes(self):
        
        data = []
        for i in range(len(self.label_interest)):
            summary = {
                'Param': self.label_interest[i],
                'Mean': np.mean(self.chains_flat[:,i]),
                'Std': np.std(self.chains_flat[:,i])
            }
            data.append(summary)
        df = pd.DataFrame(data).round(2)
        print(df)
        
        return df
    
    '''ARVIZ TOOLS'''

    def arviz_posterior_interest_plot(self, hdi = 0.96):
        
        az.style.use(["arviz-darkgrid", "arviz-cyanish"])
        fontsize_axes= 12
        fontsize_title = 15
        
        fig = az.plot_posterior(self.dataset,var_names=self.label_interest[0], 
                          kind='hist', hdi_prob =hdi, bins = 100)
        plt.suptitle("Posterior sampling of the parameter:", fontsize=fontsize_title)
        plt.xlabel('Parameter Value',fontsize=fontsize_axes)
        plt.ylabel('Frequency',fontsize=fontsize_axes)
        
        az.plot_posterior(self.dataset,var_names=self.label_interest[1], 
                          kind='hist', hdi_prob = hdi, bins = 100)
        plt.suptitle("Posterior sampling of the parameter:", fontsize=fontsize_title)
        plt.xlabel('Parameter Value',fontsize=fontsize_axes)
        plt.ylabel('Frequency',fontsize=fontsize_axes)
        
        az.plot_posterior(self.dataset,var_names=self.label_interest[2], 
                          kind='hist', hdi_prob = hdi, bins = 100)
        plt.suptitle("Posterior sampling of the parameter:", fontsize=fontsize_title)
        plt.xlabel('Parameter Value',fontsize=fontsize_axes)
        plt.ylabel('Frequency',fontsize=fontsize_axes)
        
        az.plot_posterior(self.dataset,var_names=self.label_interest[3], 
                          kind='hist', hdi_prob = hdi, bins = 100)
        plt.suptitle("Posterior sampling of the parameter:", fontsize=fontsize_title)
        plt.xlabel('Parameter Value',fontsize=fontsize_axes)
        plt.ylabel('Frequency',fontsize=fontsize_axes)

    def arviz_trace(self):
        az.plot_trace(self.dataset,var_names=self.label_interest)

    def arviz_plot_autocorr(self):
        az.plot_autocorr(self.dataset,var_names=self.label_interest, combined=True, max_lag=1000)
    
    def arviz_ess_graph(self):
        az.plot_ess(
            self.dataset, kind="local", var_names=self.label_interest, drawstyle="steps-mid", color="k",
            linestyle="-", marker=None
            )
        
    def arviz_summary_stats(self, hdi = 0.96):
        self.az_sum_stats = az.summary(self.dataset,kind = 'stats',hdi_prob=hdi,
                                       var_names=self.label_interest,round_to=2)
        print("Stats")
        print(self.az_sum_stats)