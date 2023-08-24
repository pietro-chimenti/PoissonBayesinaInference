# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 23:14:41 2023

@author: orion
"""


from Models import SignalAndNoise as SN


def main():

    ov = 2
    
    test = SN.SignalAndNoise(ov)

    prior_test1 = test.log_prior_off_on_gamma(mu_on=2, mu_off=2, alpha=2, beta=1)
    prior_test2 = test.log_prior_off_gamma(mu=4, alpha=2, beta=1)
    print(prior_test1,prior_test2)
    
    prior_test3 = test.log_prior_off_jeffrey(mu = 2)
    prior_test4 = test.log_prior_off_on_jeffrey(mu_on = 1, mu_off=1)
    print(prior_test3,prior_test4)

    data = 10
    like1 = test.log_like_off(mu=4, data=data)
    like2 = test.log_like_off_on(mu_on=2, mu_off=2, data=data)
    print(like1,like2)
    
if __name__ == "__main__":
    main()