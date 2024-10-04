import numpy
import math
import random

N_total = 1000
N_poor = 800
wealth_ratio = 0.1
error1 = 0.1
error2 = math.inf
R = 100
beta = 0.9

tau = [0] * R

for i in range(R):
    #generate uniform lifetimes for the entire population
    lifetimes = numpy.random.uniform(0, 1, N_total)
    lifetimes = numpy.sort(lifetimes)
    
    #savings vector of 1s for all individuals
    savings = numpy.ones(N_total)
    
    #randomly assign wealth_ratio to the N_poor group using vectorized operation
    poor = random.sample(range(N_total), N_poor) 
    savings[poor] = wealth_ratio
    
    #cumulative sum of savings for denominator
    cumulative_savings = numpy.cumsum(savings)
    
    u_k = lifetimes[:-1]  # everything but the last one
    u_k_plus_1 = lifetimes[1:]  # everything but the first one
    
    total_cumulative = cumulative_savings[-1]  # last in cumulative matrix
    fnct = 1 - cumulative_savings[:-1] / total_cumulative

    left_expr = (1 - u_k_plus_1) / fnct < 1 - error1
    right_expr = (1 - u_k) / fnct > 1 + error2

    if numpy.all(left_expr == False):
        k_left = math.inf
    else:
        k_left = numpy.where(left_expr)[0][0]
    if numpy.all(right_expr == False):
        k_right = math.inf
    else:
        k_right = numpy.where(right_expr)[0][0]
    
    k = min(k_left, k_right)
    if k == math.inf:
        tau[i] = 1
    else:
        if right_expr[k] == True:
            tau[i] = lifetimes[k]
        else:
            tau[i] = 1 - (1 - error1) * fnct[k]

avg_tau = sum(tau) / len(tau)
print (avg_tau)
time_result = tau #convert from uniform to lifetimes
quantile_tau = numpy.quantile(tau, 1 - beta)  # caluclate quantiles

# print ("Average amount of time:",time_result)
# print("1 - beta quantiles is:", quantile_tau)
