# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 23:14:41 2023

@author: orion
"""

import matplotlib.pyplot as plt
from Models import SignalAndNoise as SN
import numpy as np
import arviz as az

def main():
    
    data_off = np.random.randint(5, size=15)
    data_on = np.random.randint(7, size=15) 

    #test1 = SN.SignalAndNoise(data_off,data_on ,'jeffrey','gamma',mean_on=4,std_on=2)
    test1 = SN.SignalAndNoise(data_off,data_on ,'uniform','uniform')
    samples = test1.run(samples=10, burn_in= 10)
    
    print(samples.shape)

    fig, ax = plt.subplots(1, 1)
    ax.plot(samples[1 ,:,0], label="Traço da Simulação")
    plt.title("Traço da Simulação " r"$ \mu_{off} $")
    plt.show()
    
    plt.hist(samples[0,:, 0], 100, color="k", histtype="step")
    plt.title("Amostragem MCMC para " r"$ \mu_{off} $")
    plt.xlabel(r"$\mu_{off}$")
    plt.ylabel(r"$p(\mu_{off})$")
    plt.gca().set_yticks([]);
    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(samples[0,:, 1], label="Traço da Simulação")
    plt.title("Traço da Simulação " r"$ \mu_{on} $")
    plt.show()
    
    plt.hist(samples[0,:, 1], 100, color="k", histtype="step")
    plt.title("Amostragem MCMC para " r"$ \mu_{on} $")
    plt.xlabel(r"$\mu_{on}$")
    plt.ylabel(r"$p(\mu_{on})$")
    plt.gca().set_yticks([]);
    plt.show()
    
    
if __name__ == "__main__":
    main()