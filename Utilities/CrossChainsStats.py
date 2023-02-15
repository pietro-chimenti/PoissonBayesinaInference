# -*- coding: utf-8 -*-
import numpy as np

def CrossChainMean( chains, f, skip, **kwargs ):
    values = np.zeros(len(chains))
    for i in range(len(chains)):
        values[i]= f( chains[i][skip:], **kwargs )
    return np.mean(values), np.std( values, ddof = 1 )/np.sqrt(len(values))

def getIndex( chain, index ):
    return chain[:,index]

# Statistical functions of a single chain
def ChainFuncMean( chain, func ):
    return np.mean(func(chain))

def ChainFuncVar( chain, func ):
    return np.var(func(chain))

def ChainFuncQuantile( chain, quantis, func):
    return np.quantile(func(chain),quantis)

def ChainFuncFraction( chain, score, func ):
    return float((func(chain)<score).sum())/len(chain)

def ChainFuncCorr( chain, func1, func2 ):
    funcsChain = np.column_stack((func1(chain),func2(chain)))
    return np.corrcoef(funcsChain.T)[0,1]

# cross chain statistical function by index of chain elements

def CrossChainMeanIndex(chain_list, skip = 0, index = 0):
    return CrossChainMean(chain_list, ChainFuncMean, skip=skip, func = lambda x : getIndex(x, index) )

def CrossChainVarIndex(chain_list, skip = 0, index = 0):
    return CrossChainMean(chain_list, ChainFuncVar, skip=skip, func = lambda x : getIndex(x, index) )

def CrossChainQuantileIndex(chain_list, quantis, skip = 0, index = 0):
    return CrossChainMean(chain_list, ChainFuncQuantile, skip=skip, quantis = quantis, func = lambda x : getIndex(x, index) )

def CrossChainFractionIndex(chain_list, score, skip = 0, index = 0):
    return CrossChainMean(chain_list, ChainFuncFraction, skip=skip, score = score, func = lambda x : getIndex(x, index) )

# The Potential Scale Reduction Factor from Gelman and Rubin 1992 (for an arbitrary function)


def FuncPSRF( chain_list, func, skip=0):
    funcChain = []
    for i in range(len(chain_list)):
        chain = (chain_list[i])[skip:]
        funcChain.append(func(chain))
    n = len(funcChain[0])
    m = len(funcChain)
    meanChain = []
    varChain  = []
    for i in funcChain:
        meanChain.append(np.mean(i))
        varChain.append(np.var(i,ddof=1))
    B = n * np.var(np.array(meanChain),ddof=1) 
    W = np.mean(np.array(varChain))
    S2_plus = (((n-1)/n) * W) + (B/n)
    V = S2_plus + (B/(n*m))
    PSRF = V / W
    return PSRF

def PSRFIndex( chain_list, index = 0, skip = 0):
    return FuncPSRF( chain_list, func = lambda x : getIndex(x, index), skip = skip)