# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 23:14:41 2023

@author: orion
"""


from scipy.stats import poisson
from Models import SignalAndNoise as SN
import matplotlib.pyplot as plt


def main():    

    sample = [3]
        
    #sn_model = SN.SignalAndNoise(observed_values = sample)
    #bp_model = BP.BasicPoisson( observed_value=ov, prior='jeffreys' )
    sn_model = SN.SignalAndNoise( observed_value=sample ,prior='gamma', mean = 1, std = 1)
    
    sn_model.run( samples = 2000, seed = 42 )

    print("done!")
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(sample[:, 0], label="Dados Simulação")
    plt.title("Simulação Cadeia de Markov")
    plt.xlabel(r'$\lambda$')
    plt.ylabel("p.d.f.")
    plt.show()
if __name__ == "__main__":
    main()