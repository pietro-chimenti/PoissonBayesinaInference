#!/usr/bin/env python
"""This module tests the bayesian inference of the "basic" poissonian model.
"""

import sys
import matplotlib.pyplot as plt

from Models import BasicPoisson as BP

def main():  
    
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
        
    print("Input number of events:")
    ev_string = input()
    try:
        ev = int(ev_string)
    except ValueError:
        print("Value Error: must be an integer!")
        sys.exit(0)    
    if ev < 0 :
        print("Observed value must not be negative!")
        sys.exit(0)

    #rodadando o código
    model = BP.BasicPoisson(observed_value=ov, events_number = ev ,prior='jeffreys')
    x = model.interval
    y = model.distribution
    
    #gráfico
    plt.plot(x, y)
    plt.title("Posterior Distribution")
    plt.xlabel("mu")
    plt.ylabel("p.d.f.")
    plt.grid()
    plt.show()  
    
    #tabela de dados
    model.data_summarry()
    
    #calculo de probabilidade
    model.probability_calculation()

if __name__ == "__main__":
    main()
