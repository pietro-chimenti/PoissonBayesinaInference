# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:07:42 2024

@author: orion
"""

import matplotlib.pyplot as plt
from Models import Signal_and_Hierachical_Noise as SHN
import numpy as np
import arviz as az 
from scipy.stats import poisson
import pandas as pd

data_off =[19,21,23]
data_on = [30,31]
p=[-3,5,2,1,2,3,1,5,3]

test1 = SHN.Signal_and_Hierachical_Noise(data_off,data_on)

samples = 1500
burn_in = 100
nwalkers = 50
n_chains= 5

chain = test1.run(samples=samples, burn_in= burn_in,n_chains = n_chains, nwalkers= nwalkers)


#%%

plt.hist(chain[:, 0], 100, color="k", histtype="step")
plt.title('Signal Posterior')
plt.gca().set_yticks([])

#%%
array_chain1 = np.array(chain[:,1])

limite_inferior = np.percentile(array_chain1, 0)
limite_superior = np.percentile(array_chain1, 95)
dados_sem_outliers1 = array_chain1[(array_chain1 >= limite_inferior) & (array_chain1 <= limite_superior)]

plt.hist(dados_sem_outliers1,100, color="r", histtype="step")
plt.title(r"$\mu_{noise}$ Posterior" )
plt.show()
#%%
array_chain2 = np.array(chain[:,2])

limite_inferior = np.percentile(array_chain2, 0)
limite_superior = np.percentile(array_chain2, 95)
dados_sem_outliers2 = array_chain2[(array_chain2 >= limite_inferior) & (array_chain2 <= limite_superior)]

plt.hist(dados_sem_outliers2,100, color="r", histtype="step")
plt.title(r"$\sigma_{noise}$ Posterior" )
plt.show()
