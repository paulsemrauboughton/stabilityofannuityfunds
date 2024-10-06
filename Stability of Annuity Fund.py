from numba import jit
import numpy
import time

N_total = 1000
N_poor = 800
wealth_ratio = 0.1
error1 = 0.1
error2 = numpy.inf
R = 10000
beta = 0.9

@jit(nopython=True)
def montecarlo(N_total, N_poor, wealth_ratio, error1, error2, R, beta):   
    tau = numpy.zeros(R)
    
    for i in range(R):
        
        savings = numpy.ones(N_total)
        #generate uniform lifetimes for the entire population
        lifetimes = numpy.random.uniform(0, 1, N_total)
        lifetimes = numpy.sort(lifetimes)
        
        #randomly assign wealth_ratio to the N_poor group using vectorized operation
        poor = numpy.random.choice(N_total, N_poor, replace=False)
        savings[poor] = wealth_ratio
        
        #cumulative sum of savings for denominator
        cumulative_savings = numpy.cumsum(savings)
        
        u_k = lifetimes[:-1]  # everything but the last one
        u_k_plus_1 = lifetimes[1:]  # everything but the first one
        
        total_cumulative = cumulative_savings[-1]  # last in cumulative matrix
        fnct = 1 - cumulative_savings[:-1] / total_cumulative
    
        left_expr = (1 - u_k_plus_1) / fnct < 1 - error1
        right_expr = (1 - u_k) / fnct > 1 + error2
        
        k_left = numpy.where(left_expr)[0]
        k_right = numpy.where(right_expr)[0]
        
        k_left = k_left[0] if k_left.size > 0 else numpy.inf
        k_right= k_right[0] if k_right.size > 0 else numpy.inf
        
        k = min(k_left, k_right)
        if k == numpy.inf:
            tau[i] = 1
        else: 
            if right_expr[int(k)] == True:
                tau[i] = lifetimes[int(k)]
            else:
                tau[i] = 1 - (1 - error1) * fnct[int(k)]
    return tau

start_time = time.time()   #for optimisation purposes 
tau = montecarlo(N_total, N_poor, wealth_ratio, error1, error2, R, beta)
end_time = time.time()
time_taken = end_time - start_time

quantile_tau = numpy.quantile(tau, 1 - beta)

#Data taken from England & Wales 2021 Human Mortality Database.
ages = numpy.array([70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 
                  91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
lx = numpy.array([82498, 81154, 79751, 78249, 76639, 74753, 72680, 70522, 68161, 65534, 62682, 59712, 56638, 
                53340, 49847, 46171, 42356, 38426, 34437, 30386, 26494, 22664, 19025, 15601, 12509, 9788, 
                7479, 5558, 4010, 2804, 1897, 1241, 783, 477, 280, 158, 86, 45, 23, 11, 5])
l0 = lx[0]
cdf = 1 - (lx / l0)
age_at_tau = numpy.interp(quantile_tau, cdf, ages)
print ("Maximum time stable:", age_at_tau-70)
print ("Simulation time taken:",time_taken,"seconds")
