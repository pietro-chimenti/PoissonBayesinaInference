# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 23:14:41 2023

@author: orion
"""
# %% Import Libraries 
import matplotlib.pyplot as plt
from Models import SignalAndNoise as SN
import numpy as np
import arviz as az
from scipy.stats import poisson

labels = ["mu_off","mu_on"]
az.style.use("arviz-darkgrid")

# %% Run Test 1
"""Ideal Situation: Data from Poisson Sample; Flat Prior; 
High iterarions numbers; High burn-in, High Data Number"""

#Inputs
samples = 5000
burn_in = 500
nwalkers = 50
n_chains= 5

#Data Draw
data_off = poisson.rvs(mu=4,size = 100)
data_on = poisson.rvs(mu=7,size = 100) 

#Run Chains
test1 = SN.SignalAndNoise(data_off,data_on ,'uniform','uniform')
test1.run(samples=samples, burn_in= burn_in,n_chains = n_chains, nwalkers= nwalkers)

# %% Analisys Test 1

#Posterior from a single chain
single_chain1 = test1.single_chain_arvis_dataset()
az.plot_posterior(single_chain1)

#Trace from a single chains
az.plot_trace(single_chain1)

#Autocorrelation from a single chain 
for i in range(2):
    axes = az.plot_autocorr(
        single_chain1,
        var_names=labels[i],
        combined = True,
        max_lag=200
    )
    fig = axes.get_figure()
    fig.suptitle("Autocorrelation Single Chain 1", fontsize=20)
    plt.show()
    
#Effective sample size plot for a single chain 
az.plot_ess(single_chain1, kind="evolution")          
            
#R_hat from chains with diff seeds
compare_chain = test1.diff_seed_arviz_dataset()
r_hat = az.rhat(compare_chain)
print("diff seeds R^:")
print(r_hat)

#Data Summaries from all chains and walkers
full_dataset = test1.full_arviz_dataset()
data_summary = az.summary(full_dataset, hdi_prob=0.95)
print("geral data summary:")
print(data_summary)

# %% Run Test 2
"""Strong Prior: Gamma Prior(strong); few data; Large Simulation Numbers"""

#Inputs
samples2 = 3000
burn_in2 = 500
nwalkers2 = 40
n_chains2= 5

#Data Draw
data_off2 = [2]
data_on2 = [4]  

#Run Chains
test2 = SN.SignalAndNoise(data_off,data_on ,'gamma','gamma',
                          mean_off=15, mean_on=18, std_off = 2, std_on=2)
test2.run(samples=samples2, burn_in= burn_in2,n_chains = n_chains2, nwalkers= nwalkers2)

# %% Analysis Test 2

#Posterior from a single chain
single_chain2 = test2.single_chain_arvis_dataset()
az.plot_posterior(single_chain2)

#Trace from a single chains
az.plot_trace(single_chain2)

#Autocorrelation from a single chain 
for i in range(2):
    axes = az.plot_autocorr(
        single_chain2,
        var_names=labels[i],
        combined = True,
        max_lag=200
    )
    fig = axes.get_figure()
    fig.suptitle("Autocorrelation Single Chain 2", fontsize=20)
    plt.show()
    
#R_hat from chains with diff seeds
compare_chain2 = test2.diff_seed_arviz_dataset()
r_hat = az.rhat(compare_chain2)
print("diff seeds R^:")
print(r_hat)

#Data Summaries from all chains and walkers
full_dataset2 = test2.full_arviz_dataset()
data_summary2 = az.summary(full_dataset2,hdi_prob=0.95)
print("geral data summary:")
print(data_summary2)

# %% Run Test 3
"""Model Conflict: Strong prior and lot of data, with conflicting statistics"""

#Inputs
samples3 = 3000
burn_in3 = 500
nwalkers3 = 20
n_chains3= 1

#Data Draw
data_off3 = poisson.rvs(mu=4,size = 500)
data_on3 = poisson.rvs(mu=7,size = 500)  

#Run Chains
test3 = SN.SignalAndNoise(data_off3,data_on3 ,'gamma','gamma',
                          mean_off=15, mean_on=18, std_off = 2, std_on=2)
test3.run(samples=samples3, burn_in= burn_in3,n_chains = n_chains3, nwalkers= nwalkers3)


# %% Analysis Test 3

#Posterior from a single chain
single_chain3 = test3.single_chain_arvis_dataset()
az.plot_posterior(single_chain3)

#Trace from a single chains
az.plot_trace(single_chain3)

#Autocorrelation from a single chain 
for i in range(2):
    axes = az.plot_autocorr(
        single_chain3,
        var_names=labels[i],
        combined = True,
        max_lag=200
    )
    fig = axes.get_figure()
    fig.suptitle("Autocorrelation Single Chain 3", fontsize=20)
    plt.show()

#Data Summaries from a single chain
data_summary3 = az.summary(single_chain3,hdi_prob=0.95)
print("geral data summary:")
print(data_summary3)

sample_size3 = az.ess(single_chain3)
print("total eff sample size:")
print(sample_size3)

# %% Run Test 4
"""Less Info: more data_off than data_on; flat prior"""

#Inputs
samples4 = 2000
burn_in4 = 500
nwalkers4 = 20
n_chains4= 3

#Data Draw
data_off4 = poisson.rvs(mu=4,size = 500)
data_on4 = poisson.rvs(mu=7,size = 10)  

#Run Chains
test4 = SN.SignalAndNoise(data_off4,data_on4 ,'jeffrey','jeffrey')
test4.run(samples=samples4, burn_in= burn_in4,n_chains = n_chains4, nwalkers= nwalkers4)

#%% Analysis Test 4

#Posterior from a single chain
single_chain4 = test4.single_chain_arvis_dataset()
az.plot_posterior(single_chain4)

#Trace from a single chains
az.plot_trace(single_chain4)

#Autocorrelation from a single chain 
for i in range(2):
    axes = az.plot_autocorr(
        single_chain4,
        var_names=labels[i],
        combined = True,
        max_lag=200
    )
    fig = axes.get_figure()
    fig.suptitle("Autocorrelation Single Chain 4", fontsize=20)
    plt.show()

#Data Summaries from a single chain
data_summary4 = az.summary(single_chain4,hdi_prob=0.95)
print("geral data summary:")
print(data_summary4)

sample_size4 = az.ess(single_chain4)
print("total eff sample size:")
print(sample_size4)


# %% Run Test 5 
"""Data Conflict: data_off> data_on; flat prior"""

#Inputs
samples5 = 2000
burn_in5 = 500
nwalkers5 = 20
n_chains5= 3

#Data Draw
data_off5 = poisson.rvs(mu=8,size = 50)
data_on5 = poisson.rvs(mu=2,size = 50)  

#Run Chains
test5 = SN.SignalAndNoise(data_off5,data_on5 ,'jeffrey','jeffrey')
test5.run(samples=samples5, burn_in= burn_in5,n_chains = n_chains5, nwalkers= nwalkers5)

# %% Analysis Test 5

#Posterior from a single chain
single_chain5 = test5.single_chain_arvis_dataset()
az.plot_posterior(single_chain5)

#Trace from a single chains
az.plot_trace(single_chain5)

#Autocorrelation from a single chain 
for i in range(2):
    axes = az.plot_autocorr(
        single_chain5,
        var_names=labels[i],
        combined = True,
        max_lag=200
    )
    fig = axes.get_figure()
    fig.suptitle("Autocorrelation Single Chain 5", fontsize=20)
    plt.show()

#Data Summaries from a single chain
data_summary5 = az.summary(single_chain5,hdi_prob=0.95)
print("geral data summary:")
print(data_summary5)

#%% Random Test Analysis
'''Inputs'''

samples = 1000
burn_in = 500
nw = 30
n_chains= 4


data_off = np.random.randint(5, size=15)
data_on = np.random.randint(7, size=15) 

'''Run Chains'''
#test1 = SN.SignalAndNoise(data_off,data_on ,'jeffrey','gamma',mean_on=4,std_on=2)
test1 = SN.SignalAndNoise(data_off,data_on ,'uniform','uniform')
chains = test1.run(samples=samples, burn_in= burn_in,n_chains = n_chains, nwalkers= nw)

'''Total Comparging Chains and Walkers'''

azdata = test1.full_arviz_dataset()

sample_size = az.ess(azdata)
print("total eff sample size:")
print(sample_size)

r_hat = az.rhat(azdata)
print("total R^:")
print(r_hat)

data_summary = az.summary(azdata)
print("geral data summary:")
print(data_summary)


'''Comparing Different Chains (walker 0)'''

compare_chain = test1.diff_seed_arviz_dataset()

for i in range(2):
    axes = az.plot_autocorr(
        compare_chain,
        var_names=labels[i]
    )
    fig = axes.flatten()[0].get_figure()
    fig.suptitle("Autocorrelation", fontsize=30)
    plt.show()
    
"""Single Chain Analisys"""

single_chain = test1.single_chain_arvis_dataset()
az.plot_trace(single_chain)


  

    
