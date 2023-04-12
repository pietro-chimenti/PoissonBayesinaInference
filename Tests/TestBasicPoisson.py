#!/usr/bin/env python
"""This module tests the bayesian inference of the "basic" poissonian model.
"""

import sys
import matplotlib.pyplot as plt

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
    
    #constantes
    ov = 5
    credible_interval = 0.95
    prior = 'gamma'
    mean = 4
    stan_desv = 2
    
    #rodadando o código
    model = BP.BasicPoisson(observed_value=ov,prior = prior, mean = mean, std = stan_desv)
    x = model.interval
    y = model.distribution
    
    #Intervalo de Credibilidade Limite Superior
    up1, down1 = model.credible_interval(trust = credible_interval, option = 1)

    #Intervalo de Credibilidade Simetrico
    up2, down2 = model.credible_interval(trust= credible_interval, option = 2)
    
    #Intervalo de Credibilidade HDI
    up3, down3 = model.credible_interval(trust= credible_interval, option = 3)
    
    #gráfico
    plt.plot(x, y)
    plt.title("Posterior Distribution")
    plt.xlabel("mu")
    plt.ylabel("p.d.f.")
    plt.axvline(x = up1, color = 'r', label = 'Upper limit')
    plt.axvline(x = down1, color = 'r')
    plt.axvline(x = up2, color = 'g', label = 'Symmetrical')
    plt.axvline(x = down2, color = 'g')
    plt.axvline(x = up3, color = 'm', label = 'HDI')
    plt.axvline(x = down3, color = 'm')
    plt.legend()
    plt.grid()
    plt.show()  
    
    #tabela de dados
    model.data_summarry()


if __name__ == "__main__":
    main()
