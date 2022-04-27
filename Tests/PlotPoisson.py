#!/usr/bin/env python
"""This module produces plots for the bayesian inference of the poissonian model.
"""

import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt


def main():
    fig, ax = plt.subplots(1, 1)
    mu_1 = 0.6
    x_1 = np.arange(poisson.ppf(0.01, mu_1),
        poisson.ppf(0.99, mu_1))
    ax.plot(x_1, poisson.pmf(x_1, mu_1), 'bo', ms=8, label='poisson pmf - 0.6')

    mu_2 = 3.0
    x_2 = np.arange(poisson.ppf(0.01, mu_2),
        poisson.ppf(0.99, mu_2))
    ax.plot(x_2, poisson.pmf(x_2, mu_2), 'ro', ms=8, label='poisson pmf - 3.0')

    ax.legend(loc='best', frameon=False)
    plt.show()

if __name__ == "__main__":
    main()
