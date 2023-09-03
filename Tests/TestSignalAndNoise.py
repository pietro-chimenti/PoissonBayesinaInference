# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 23:14:41 2023

@author: orion
"""

import matplotlib.pyplot as plt
from Models import SignalAndNoise as SN
import numpy as np
import arviz as az
import xarray as xr
from matplotlib.axis import Axis

'''Inputs'''

samples = 1000
burn_in = 500
nw = 30
n_chains= 4
labels = ["mu_off","mu_on"]

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


  

    
