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

labels = [r"\mu_{off}",r"\mu_{on}"]
az.style.use("arviz-darkgrid")

bins = 15

# %% Run Test 1


#Inputs
samples = 100
burn_in = 50
nwalkers = 30
n_chains= 5

#Data Draw
data_off = poisson.rvs(mu=4,size = 100)
data_on = poisson.rvs(mu=7,size = 100) 

#Run Chains
test1 = SN.SignalAndNoise(data_off,data_on ,'uniform','uniform')
test1.run(samples=samples, burn_in= burn_in,n_chains = n_chains, nwalkers= nwalkers)

# %% Analisys Test 1

ds = test1.full_arviz_dataset()
single_chain = test1.single_chain_arvis_dataset()

test1.diagnose(ds, single_chain, bins = bins, title = "Test 1")

test1.statistic_error()

x, y = test1.preditive_distribution()



  

    
