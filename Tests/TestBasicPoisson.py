#!/usr/bin/env python
"""This module tests the bayesian inference of the "basic" poissonian model.
"""

import sys
import matplotlib.pyplot as plt
from scipy.stats import poisson
import numpy as np

from Models import BasicPoisson as BP

def main():  
    
#está parte está comentada para retirar o input do usario, facilitando testes
    '''
    #inserindo constantes
    print("Input the observed value:")
    ov_string = input()
    try:
        ov = int(ov_string)
    except ValueError:
        print("Value Error: must be an integer!")
            sys.exit(0)    
    if ov < 0 :
        print("Observed value must not be negative!")
        sys.exit(0)

    print('Choose the type of credible interal')
    print(' Upper Limit Interval - 1\n Symmetrical Interval - 2 \n High Density Interval - 3')
    print('Insert the number of the option:')
    option = input()
    print('Insert the Credible Interval %:')
    inp = input()
    trust = int(inp)/100
    '''    
    
    #SAMPLER
    size = 1000
    mu=4
    
    #Holdout Cross Verification 50%
    sample = poisson.rvs(mu,size = size) 
    
    ov = sample[:size//2]     #Model Data
    verify = sample[size//2:] #Verification Data
    
    #constantes
    credible_interval = 0.95
    prior = 'jeffreys'
    mean = 10
    stan_desv = 2
    
    #rodadando o código
    model = BP.BasicPoisson(observed_value=ov,prior = prior, mean = mean, std = stan_desv)

    x = model.interval
    y = model.distribution
    
    #Intervalo de Credibilidade Limite Superior
    up1= model.credible_interval(trust = credible_interval, option = 1)

    #Intervalo de Credibilidade Simetrico
    up2, down2 = model.credible_interval(trust= credible_interval, option = 2)
    
    #Intervalo de Credibilidade HDI
    up3, down3 = model.credible_interval(trust= credible_interval, option = 3)

    #gráfico
    plt.plot(x, y)
    plt.title("Posterior Parameter Distribution")
    plt.xlabel("mu")
    plt.ylabel("p.d.f.")
    plt.axvline(x = up1, color = 'r', label = 'Upper limit')
    plt.axvline(x = up2, color = 'g', label = 'Symmetrical')
    plt.axvline(x = down2, color = 'g')
    plt.axvline(x = up3, color = 'm', label = 'HDI')
    plt.axvline(x = down3, color = 'm')
    plt.legend()
    plt.show()  

    #tabela de dados
    model.data_summarry()
    
    #posterior prediction
    plt.style.use('seaborn-whitegrid')
    
    #Verification Data
    plt.hist(ov,bins=np.arange(max(verify)+1), density = True, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5, label='Observed Data')
    
    postx, posty, p_value = model.predictive_posterior_run()
    plt.scatter(postx,posty,color='r', label = 'Post. Pred. Dist.')
   
    plt.title("Posterior Predictive Distribution")
    plt.ylabel("p.d.f.")
    plt.xlabel('Data')
    plt.legend()
    plt.show()
    
    #prints
    '''
    print(f'the observed data is:{ov}')
    print(f'the p-value list is:{p_value}')
    '''
    
if __name__ == "__main__":
    main()
