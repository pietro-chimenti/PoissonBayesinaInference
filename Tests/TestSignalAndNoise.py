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
import pandas as pd

labels = ["mu_off","mu_on"]
az.style.use("arviz-darkgrid")

# %% Run Test 1


#Inputs
samples = 500
burn_in = 500
nwalkers = 100
n_chains= 8

#Data Draw
data_off = poisson.rvs(mu=4,size = 100)
data_on = poisson.rvs(mu=7,size = 100) 

#Run Chains
test1 = SN.SignalAndNoise(data_off,data_on ,'uniform','uniform')
test1.run(samples=samples, burn_in= burn_in,n_chains = n_chains, nwalkers= nwalkers)

# %% Analisys Test 1

ds = test1.full_arviz_dataset()
single_chain = test1.single_chain_arvis_dataset()
az.plot_trace(single_chain)
plt.show()
bins = 15

ess = az.ess(ds)

r_hat = az.rhat(ds)

mcse = az.mcse(ds)

summary = {
    'Stat': ['ess off','ess on','rhat off','rhat on', 'msce off', 'msce on'],
    'Mean': [float(ess['mu_off'].mean()),float(ess['mu_on'].mean()),float(r_hat['mu_off'].mean()),float(r_hat['mu_on'].mean()),float(mcse['mu_off'].mean()),float(mcse['mu_on'].mean())],
    'Std': [float(ess['mu_off'].std()),float(ess['mu_on'].std()),float(r_hat['mu_off'].std()),float(r_hat['mu_on'].std()),float(mcse['mu_off'].std()),float(mcse['mu_on'].std())]
}

df = pd.DataFrame(summary).round(5)
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')
tabela = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=['#f2f2f2']*len(df.columns))
tabela.auto_set_font_size(False)
tabela.set_fontsize(12)
tabela.scale(1.5, 1.5)
plt.title("Test 1 - 500 Samples")
plt.show()

fig, axs = plt.subplots(nrows=2, ncols=3)

axs[0, 0].hist(ess['mu_off'], bins = bins, histtype='stepfilled', facecolor='g',
               alpha=0.75)
axs[0, 0].set_title('effect sample size mu_off')

axs[1, 0].hist(ess['mu_on'], bins = bins, histtype='stepfilled', facecolor='g',
               alpha=0.75)
axs[1, 0].set_title('effect sample size mu_on')


axs[0, 1].hist(r_hat['mu_off'], bins=bins, histtype='stepfilled', facecolor='b',
               alpha=0.75)
axs[0, 1].set_title('r^ mu_off')

axs[1, 1].hist(r_hat['mu_on'], bins=bins, histtype='stepfilled', facecolor='b',
               alpha=0.75)
axs[1, 1].set_title('r^ mu_on')

axs[0, 2].hist(mcse['mu_off'], bins=bins, histtype='stepfilled', facecolor='r',
               alpha=0.75)
axs[0, 2].set_title('MC Std Error mu_off')

axs[1, 2].hist(mcse['mu_on'], bins=bins, histtype='stepfilled', facecolor='r',
               alpha=0.75)
axs[1, 2].set_title('MC Std Error mu_on')

plt.suptitle('Test 1 - 500 Samples', fontsize=18)
plt.show()

# %% Run Test 2


#Inputs
samples2 = 1000
burn_in2 = 500
nwalkers2 = 100
n_chains2= 8

#Data Draw
data_off = poisson.rvs(mu=4,size = 100)
data_on = poisson.rvs(mu=7,size = 100)

#Run Chains
test2 = SN.SignalAndNoise(data_off,data_on ,'uniform','uniform')
                          
test2.run(samples=samples2, burn_in= burn_in2,n_chains = n_chains2, nwalkers= nwalkers2)

# %% Analysis Test 2

ds = test2.full_arviz_dataset()
single_chain = test2.single_chain_arvis_dataset()
az.plot_trace(single_chain)
plt.show()
bins = 15

ess = az.ess(ds)

r_hat = az.rhat(ds)

mcse = az.mcse(ds)

summary = {
    'Stat': ['ess off','ess on','rhat off','rhat on', 'msce off', 'msce on'],
    'Mean': [float(ess['mu_off'].mean()),float(ess['mu_on'].mean()),float(r_hat['mu_off'].mean()),float(r_hat['mu_on'].mean()),float(mcse['mu_off'].mean()),float(mcse['mu_on'].mean())],
    'Std': [float(ess['mu_off'].std()),float(ess['mu_on'].std()),float(r_hat['mu_off'].std()),float(r_hat['mu_on'].std()),float(mcse['mu_off'].std()),float(mcse['mu_on'].std())]
}
df = pd.DataFrame(summary).round(5)
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')
tabela = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=['#f2f2f2']*len(df.columns))
tabela.auto_set_font_size(False)
tabela.set_fontsize(12)
tabela.scale(1.5, 1.5)
plt.title("Test 2 - 1000 Samples")
plt.show()

fig, axs = plt.subplots(nrows=2, ncols=3)

axs[0, 0].hist(ess['mu_off'], bins = bins, histtype='stepfilled', facecolor='g',
               alpha=0.75)
axs[0, 0].set_title('effect sample size mu_off')

axs[1, 0].hist(ess['mu_on'], bins = bins, histtype='stepfilled', facecolor='g',
               alpha=0.75)
axs[1, 0].set_title('effect sample size mu_on')


axs[0, 1].hist(r_hat['mu_off'], bins=bins, histtype='stepfilled', facecolor='b',
               alpha=0.75)
axs[0, 1].set_title('r^ mu_off')

axs[1, 1].hist(r_hat['mu_on'], bins=bins, histtype='stepfilled', facecolor='b',
               alpha=0.75)
axs[1, 1].set_title('r^ mu_on')

axs[0, 2].hist(mcse['mu_off'], bins=bins, histtype='stepfilled', facecolor='r',
               alpha=0.75)
axs[0, 2].set_title('MC Std Error mu_off')

axs[1, 2].hist(mcse['mu_on'], bins=bins, histtype='stepfilled', facecolor='r',
               alpha=0.75)
axs[1, 2].set_title('MC Std Error mu_on')

plt.suptitle('Test 2 - 1000 Samples', fontsize=18)
plt.show()

# %% Run Test 4


#Inputs
samples3 = 3000
burn_in3 = 500
nwalkers3 = 100
n_chains3= 8
#Data Draw
data_off3 = poisson.rvs(mu=4,size = 100)
data_on3 = poisson.rvs(mu=7,size = 100)  

#Run Chains
test3 = SN.SignalAndNoise(data_off3,data_on3 ,'uniform','uniform')
                          
test3.run(samples=samples3, burn_in= burn_in3,n_chains = n_chains3, nwalkers= nwalkers3)


# %% Analysis Test 4

ds = test3.full_arviz_dataset()
single_chain = test3.single_chain_arvis_dataset()
az.plot_trace(single_chain)
plt.show()
bins = 15

ess = az.ess(ds)

r_hat = az.rhat(ds)

mcse = az.mcse(ds)

summary = {
    'Stat': ['ess off','ess on','rhat off','rhat on', 'msce off', 'msce on'],
    'Mean': [float(ess['mu_off'].mean()),float(ess['mu_on'].mean()),float(r_hat['mu_off'].mean()),float(r_hat['mu_on'].mean()),float(mcse['mu_off'].mean()),float(mcse['mu_on'].mean())],
    'Std': [float(ess['mu_off'].std()),float(ess['mu_on'].std()),float(r_hat['mu_off'].std()),float(r_hat['mu_on'].std()),float(mcse['mu_off'].std()),float(mcse['mu_on'].std())]
}
df = pd.DataFrame(summary).round(5)
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')
tabela = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=['#f2f2f2']*len(df.columns))
tabela.auto_set_font_size(False)
tabela.set_fontsize(12)
tabela.scale(1.5, 1.5)
plt.title("Test 4 - 3000 Samples")
plt.show()

fig, axs = plt.subplots(nrows=2, ncols=3)

axs[0, 0].hist(ess['mu_off'], bins = bins, histtype='stepfilled', facecolor='g',
               alpha=0.75)
axs[0, 0].set_title('effect sample size mu_off')

axs[1, 0].hist(ess['mu_on'], bins = bins, histtype='stepfilled', facecolor='g',
               alpha=0.75)
axs[1, 0].set_title('effect sample size mu_on')


axs[0, 1].hist(r_hat['mu_off'], bins=bins, histtype='stepfilled', facecolor='b',
               alpha=0.75)
axs[0, 1].set_title('r^ mu_off')

axs[1, 1].hist(r_hat['mu_on'], bins=bins, histtype='stepfilled', facecolor='b',
               alpha=0.75)
axs[1, 1].set_title('r^ mu_on')

axs[0, 2].hist(mcse['mu_off'], bins=bins, histtype='stepfilled', facecolor='r',
               alpha=0.75)
axs[0, 2].set_title('MC Std Error mu_off')

axs[1, 2].hist(mcse['mu_on'], bins=bins, histtype='stepfilled', facecolor='r',
               alpha=0.75)
axs[1, 2].set_title('MC Std Error mu_on')

plt.suptitle('Test 4 - 3000 Samples', fontsize=18)
plt.show()

# %% Run Test 3


#Inputs
samples4 = 2000
burn_in4 = 500
nwalkers4 = 100
n_chains4= 8

#Data Draw
data_off4 = poisson.rvs(mu=4,size = 100)
data_on4 = poisson.rvs(mu=7,size = 100)  

#Run Chains
test4 = SN.SignalAndNoise(data_off4,data_on4 ,'uniform','uniform')
test4.run(samples=samples4, burn_in= burn_in4,n_chains = n_chains4, nwalkers= nwalkers4)

#%% Analysis Test 3

ds = test4.full_arviz_dataset()
single_chain = test4.single_chain_arvis_dataset()
az.plot_trace(single_chain)
plt.show()
bins = 15

ess = az.ess(ds)

r_hat = az.rhat(ds)

mcse = az.mcse(ds)

summary = {
    'Stat': ['ess off','ess on','rhat off','rhat on', 'msce off', 'msce on'],
    'Mean': [float(ess['mu_off'].mean()),float(ess['mu_on'].mean()),float(r_hat['mu_off'].mean()),float(r_hat['mu_on'].mean()),float(mcse['mu_off'].mean()),float(mcse['mu_on'].mean())],
    'Std': [float(ess['mu_off'].std()),float(ess['mu_on'].std()),float(r_hat['mu_off'].std()),float(r_hat['mu_on'].std()),float(mcse['mu_off'].std()),float(mcse['mu_on'].std())]
}
df = pd.DataFrame(summary).round(5)
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')
tabela = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=['#f2f2f2']*len(df.columns))
tabela.auto_set_font_size(False)
tabela.set_fontsize(12)
tabela.scale(1.5, 1.5)
plt.title("Test 3 - 2000 Samples")
plt.show()

fig, axs = plt.subplots(nrows=2, ncols=3)

axs[0, 0].hist(ess['mu_off'], bins = bins, histtype='stepfilled', facecolor='g',
               alpha=0.75)
axs[0, 0].set_title('effect sample size mu_off')

axs[1, 0].hist(ess['mu_on'], bins = bins, histtype='stepfilled', facecolor='g',
               alpha=0.75)
axs[1, 0].set_title('effect sample size mu_on')


axs[0, 1].hist(r_hat['mu_off'], bins=bins, histtype='stepfilled', facecolor='b',
               alpha=0.75)
axs[0, 1].set_title('r^ mu_off')

axs[1, 1].hist(r_hat['mu_on'], bins=bins, histtype='stepfilled', facecolor='b',
               alpha=0.75)
axs[1, 1].set_title('r^ mu_on')

axs[0, 2].hist(mcse['mu_off'], bins=bins, histtype='stepfilled', facecolor='r',
               alpha=0.75)
axs[0, 2].set_title('MC Std Error mu_off')

axs[1, 2].hist(mcse['mu_on'], bins=bins, histtype='stepfilled', facecolor='r',
               alpha=0.75)
axs[1, 2].set_title('MC Std Error mu_on')

plt.suptitle('Test 3 - 2000 Samples', fontsize=18)
plt.show()


# %% Run Test 5 
"""Data Conflict: data_off> data_on; flat prior"""

#Inputs
samples5 = 4000
burn_in5 = 500
nwalkers5 = 100
n_chains5= 8

#Data Draw
data_off5 = poisson.rvs(mu=4,size = 100)
data_on5 = poisson.rvs(mu=7,size = 100)  

#Run Chains
test5 = SN.SignalAndNoise(data_off5,data_on5 ,'uniform','uniform')
test5.run(samples=samples5, burn_in= burn_in5,n_chains = n_chains5, nwalkers= nwalkers5)

# %% Analysis Test 5

ds = test5.full_arviz_dataset()
single_chain = test5.single_chain_arvis_dataset()
az.plot_trace(single_chain)
plt.show()
bins = 15

ess = az.ess(ds)

r_hat = az.rhat(ds)

mcse = az.mcse(ds)

summary = {
    'Stat': ['ess off','ess on','rhat off','rhat on', 'msce off', 'msce on'],
    'Mean': [float(ess['mu_off'].mean()),float(ess['mu_on'].mean()),float(r_hat['mu_off'].mean()),float(r_hat['mu_on'].mean()),float(mcse['mu_off'].mean()),float(mcse['mu_on'].mean())],
    'Std': [float(ess['mu_off'].std()),float(ess['mu_on'].std()),float(r_hat['mu_off'].std()),float(r_hat['mu_on'].std()),float(mcse['mu_off'].std()),float(mcse['mu_on'].std())]
}
df = pd.DataFrame(summary).round(5)
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')
tabela = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=['#f2f2f2']*len(df.columns))
tabela.auto_set_font_size(False)
tabela.set_fontsize(12)
tabela.scale(1.5, 1.5)
plt.title("Test 5 - 4000 Samples")
plt.show()

fig, axs = plt.subplots(nrows=2, ncols=3)

axs[0, 0].hist(ess['mu_off'], bins = bins, histtype='stepfilled', facecolor='g',
               alpha=0.75)
axs[0, 0].set_title('effect sample size mu_off')

axs[1, 0].hist(ess['mu_on'], bins = bins, histtype='stepfilled', facecolor='g',
               alpha=0.75)
axs[1, 0].set_title('effect sample size mu_on')


axs[0, 1].hist(r_hat['mu_off'], bins=bins, histtype='stepfilled', facecolor='b',
               alpha=0.75)
axs[0, 1].set_title('r^ mu_off')

axs[1, 1].hist(r_hat['mu_on'], bins=bins, histtype='stepfilled', facecolor='b',
               alpha=0.75)
axs[1, 1].set_title('r^ mu_on')

axs[0, 2].hist(mcse['mu_off'], bins=bins, histtype='stepfilled', facecolor='r',
               alpha=0.75)
axs[0, 2].set_title('MC Std Error mu_off')

axs[1, 2].hist(mcse['mu_on'], bins=bins, histtype='stepfilled', facecolor='r',
               alpha=0.75)
axs[1, 2].set_title('MC Std Error mu_on')

plt.suptitle('Test 5 - 4000 Samples', fontsize=18)
plt.show()


#%% Run Test 6 
'''Few burn_in'''
#Inputs
samples6 = 2000
burn_in6 = 100
nwalkers6 = 100
n_chains6= 8

#Data Draw
data_off6 = poisson.rvs(mu=6,size = 50)
data_on6 = poisson.rvs(mu=8,size = 50)  

#Run Chains
test6 = SN.SignalAndNoise(data_off6,data_on6 ,'uniform','uniform')
test6.run(samples=samples6, burn_in= burn_in6,n_chains = n_chains6, nwalkers= nwalkers6)

#%% Analysis Test 6

ds = test6.full_arviz_dataset()
single_chain = test6.single_chain_arvis_dataset()
az.plot_trace(single_chain)
plt.show()
bins = 15

ess = az.ess(ds)
print("effective sample size:")
print(ess)


plt.hist(ess['mu_off'],bins=bins, edgecolor='k')
plt.xlabel('mu_off')
plt.ylabel('Frequência')
plt.title('effective sample size mu_off')
plt.grid(True)
plt.show()

plt.hist(ess['mu_on'],bins=bins, edgecolor='k')
plt.xlabel('mu_on')
plt.ylabel('Frequência')
plt.title('effective sample size mu_on ')
plt.grid(True)
plt.show()

r_hat = az.rhat(ds)
print("r^:")
print(r_hat)

plt.hist(r_hat['mu_off'],bins=bins, edgecolor='k')
plt.xlabel('mu_off')
plt.ylabel('Frequência')
plt.title('r^ mu_off')
plt.grid(True)
plt.show()

plt.hist(r_hat['mu_on'],bins=bins, edgecolor='k')
plt.xlabel('mu_on')
plt.ylabel('Frequência')
plt.title('r^ mu_on')
plt.grid(True)
plt.show()

mcse = az.mcse(ds)
print("Markov Chain Standard Error statistic:")
print(mcse)

plt.hist(mcse['mu_off'],bins=bins, edgecolor='k')
plt.xlabel('mu_off')
plt.ylabel('Frequência')
plt.title('Markov Chain Standard Error statistic mu_off')
plt.grid(True)
plt.show()

plt.hist(mcse['mu_on'],bins=bins, edgecolor='k')
plt.xlabel('mu_on')
plt.ylabel('Frequência')
plt.title('Markov Chain Standard Error statistic mu_on')
plt.grid(True)
plt.show()

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


  

    
