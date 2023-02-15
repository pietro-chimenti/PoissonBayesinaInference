# -*- coding: utf-8 -*-
""" We test the CrossChianMean utility on gaussian random numbers """


import numpy as np
from Utilities import CrossChainsStats as ccs

# now let´s generate normal covariate random "chains"
chain_list = []

mean = [0,0]
cov  = [[1,0.5],[0.5,1]]

for i in range(8):
    chain_list.append(np.random.multivariate_normal(mean, cov, size=1000000))

# now test the utility against the known values

cmean = ccs.CrossChainMean( chain_list, ccs.ChainFuncMean, skip=0, func = lambda x : ccs.getIndex(x, 0) )
np.testing.assert_array_almost_equal(cmean[0], 0, decimal = 3)
print("Mean 1 ok:")
print("{:.5f} pm {:.5f}\n".format(cmean[0],cmean[1]))

cvar = ccs.CrossChainMean(chain_list, ccs.ChainFuncVar, skip=0,  func = lambda x : ccs.getIndex(x, 0) )
np.testing.assert_array_almost_equal(cvar[0], 1, decimal = 3)
print("Var 1 ok :")
print("{:.5f} pm {:.5f}\n".format(cvar[0],cvar[1]))

cquartile = ccs.CrossChainMean(chain_list, ccs.ChainFuncQuantile, skip=0, quantis = 0.25, func = lambda x : ccs.getIndex(x, 0) )
np.testing.assert_array_almost_equal(cquartile[0], -0.674, decimal = 3)
print("First quartile 1 ok: ")
print("{:.5f} pm {:.5f}\n".format(cquartile[0],cquartile[1]))

cmean_ls = ccs.CrossChainMeanIndex(chain_list, skip = 999900, index = 0)
np.testing.assert_array_almost_equal(cmean_ls[0], 0, decimal = 1)
print("Mean 1 low stats ok:")
print("{:.3f} pm {:.3f}\n".format(cmean_ls[0],cmean_ls[1]))

cvar_ls = ccs.CrossChainVarIndex(chain_list, skip = 999900, index = 0)
np.testing.assert_array_almost_equal(cvar_ls[0], 1, decimal = 1)
print("Var 1 low stats ok :")
print("{:.3f} pm {:.3f}\n".format(cvar[0],cvar[1]))

cquartile_ls = ccs.CrossChainQuantileIndex(chain_list, quantis = 0.25, skip = 999900, index = 0)
np.testing.assert_array_almost_equal(cquartile_ls[0], -0.674, decimal = 1)
print("First quartile 1 low stats ok: ")
print("{:.3f} pm {:.3f}\n".format(cquartile[0],cquartile[1]))

cfraction = ccs.CrossChainFractionIndex(chain_list, score = 0., skip = 0, index = 0)
np.testing.assert_array_almost_equal(cfraction[0], 0.5, decimal = 3)
print("Fraction less than 0 ok: ")
print("{:.5f} pm {:.5f}\n".format(cfraction[0],cfraction[1]))

corrcoef = ccs.CrossChainMean(chain_list, ccs.ChainFuncCorr, skip=0, func1 = lambda x : ccs.getIndex(x, 0) , func2 = lambda x : ccs.getIndex(x, 1) )
np.testing.assert_array_almost_equal(corrcoef[0], 0.5, decimal = 3)
print("Correlation coefficient ok:")
print("{:.5f} pm {:.5f}\n".format(corrcoef[0],corrcoef[1]))

print("PSRF 10      sample: ", ccs.PSRFIndex(chain_list, skip = 999990))
print("PSRF 100     sample: ", ccs.PSRFIndex(chain_list, skip = 999900))
print("PSRF 1000    sample: ", ccs.PSRFIndex(chain_list, skip = 999000))
print("PSRF 10000   sample: ", ccs.PSRFIndex(chain_list, skip = 990000))
print("PSRF 100000  sample: ", ccs.PSRFIndex(chain_list, skip = 900000))
print("PSRF 1000000 sample: ", ccs.PSRFIndex(chain_list))

