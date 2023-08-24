# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 23:14:41 2023

@author: orion
"""


from Models import SignalAndNoise as SN


def main():
    
    test1 = SN.SignalAndNoise([1,2], [3,4],'jeffrey','uniform')
    post1 = test1.log_posterior(mu=[10,2])
    print(post1)
    
    test2 = SN.SignalAndNoise([1,2], [3,4],'gamma','uniform',mean_off= 2, std_off=2)
    post2 = test2.log_posterior(mu=[10,6])
    print(post2)
    
    test3 = SN.SignalAndNoise([1,2,5,8,10], [3,4,5,3],'gamma','gamma',mean_on=50, std_off=20)
    post3 = test3.log_posterior(mu=[30,21])
    print(post3)
    
    test4 = SN.SignalAndNoise([-1,2], [3,4],'jeffrey','uniform')

    post4 = test4.log_posterior(mu=[10,-12])
    print(post4)
if __name__ == "__main__":
    main()