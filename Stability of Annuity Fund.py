# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:16:39 2024

@author: paulb
"""
import numpy
import math
import random



N_total = 1000
N_poor = 800
N_rich = N_total - N_poor  
wealth_ratio = 0.3
error1 = 0.1  
error2= math.inf
certainty_beta = 0.9 
R = 2

def generate_lifetimes(N_total):
    """
    Generates the order statistics Uk from exponential distribution 
    and orders them using numpy for now (inefficient), returns them as array.
    """
    lifetimes=numpy.random.exponential(1, N_total)
    return numpy.sort(lifetimes).tolist()

def random_sample(N_total, N_poor, lifetimes):
    """
    Choose N_poor numbers at random from 1 to N_total without replacement, 
    The numbers are used to assign random lifetimes to the N_poor group,
    returns the lifetimes assigned to N_poor and N_rich.
    ( and sorts using numpy again... )
    """
    
    lifetimes_copy = lifetimes.copy()
    lifetime_of_poor = [0] * N_poor
    lifetimes_N1 = random.sample(range(N_total),N_poor)
    
    for i in range (0, len(lifetimes_N1)):
        
        lifetime_of_poor[i] =  lifetimes[lifetimes_N1[i]]
        lifetimes_copy[lifetimes_N1[i]] = 0
        
    lifetimes_copy = [x for x in lifetimes_copy if x != 0]
    lifetime_of_poor = numpy.sort(lifetime_of_poor).tolist()
    x = (lifetime_of_poor, lifetimes_copy)
    
    return x


def function(lifetimes, N_poor_lifetimes, N_rich_lifetimes, wealth_ratio,k):
    """
    This computes 1 − ˆF(F−1(v)) in the remark in simulation A.5 for
    use in each Monte-Carlo simulation, returns a number.
    """
    result = 0
    lifetimes_copy = lifetimes.copy()
    N_poor_lifetimes_copy = N_poor_lifetimes.copy()
    N_rich_lifetimes_copy = N_rich_lifetimes.copy()
    for i in range(k,len(lifetimes)):
        if lifetimes_copy[i] in N_poor_lifetimes_copy:
            N_poor_lifetimes_copy.remove(lifetimes_copy[i])
            Si = wealth_ratio
        elif lifetimes_copy[i] in N_rich_lifetimes_copy:
            N_rich_lifetimes_copy.remove(lifetimes_copy[i])
            Si = 1
        elif lifetimes_copy[i] not in N_poor_lifetimes_copy or N_rich_lifetimes_copy:
            Si = 0
        
        result = result + Si
        lifetimes_copy[i] = 0
    
    result = 1 - 1/(len(lifetimes)) * result 
    return result

def monte_carlo_simulation(N_total, N_poor, R, error1, error2, wealth_ratio):
    """
    Runs Monte-Carlo simulation R times, in order to find the first k between 0
    and N_total such that the expressions A8 and A9 are satisfied- implying that
    the annuity fund is unstable. Returns the number people that have died and
    the lifetime of the last person to die before the fund becomes unstable. 
    becomes unstable.
    """
    
    k_results=[0] * R
    tau_results= [0] * R
    
    for i in range (0,R):
        
        lifetimes = generate_lifetimes(N_total)
        x = random_sample(N_total,N_poor, lifetimes)
        N_poor_lifetimes = x[0]
        N_rich_lifetimes = x[1]
        
        # print (lifetimes)
        # print (N_poor_lifetimes)
        # print(N_rich_lifetimes)
        
        for k in range(0,N_total-1):
            
            fnct = function(lifetimes, N_poor_lifetimes, N_rich_lifetimes, wealth_ratio, k)
            u_k = lifetimes[k]
            u_k_plus_1 = lifetimes[k+1]
            
            left_expr = (1 - u_k_plus_1) / (1 - fnct)
            right_expr = (1 - u_k) / (1 - fnct)
            
            if left_expr < (1 - error1):
                k_results[i] = k
                tau_results[i] = 1 - (1 - error1) * (1 - fnct)
                break
            
            if (1 + error2) < right_expr:
                k_results[i] = k
                tau_results[i] = u_k
                break
        
    return (k_results,tau_results)

def get_time(tau_results):
    """
    Function to get Tau results (unsure of this one),
    I have used the exponential order statistic to convert 
    it to an amount of time??? 
    """
    print (tau_results)
    times = [0] * len(tau_results)
    for i in range (0, len(tau_results)):
         times[i] =  (1 - math.exp(-tau_results[i])) * 70
    return times 
            
results = monte_carlo_simulation(N_total,N_poor,R,error1,error2, wealth_ratio)
print (results)
print ("Amount of time in years for annuity to become unstable:", get_time(results[1]))