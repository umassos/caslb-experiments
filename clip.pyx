import random
import math
from scipy.optimize import linprog
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import newton
import scipy.integrate as integrate
from scipy.special import lambertw
import sys
import numpy as np
import pandas as pd
import pickle

cimport numpy as np
np.import_array()

# weighted_l1_norm computes the weighted L1 norm between two vectors.
def weighted_l1_norm(vector1, vector2, weights):
    assert vector1.shape == vector2.shape == weights.shape, "Input arrays must have the same shape."

    weighted_diff = np.abs(vector1 - vector2) * weights
    weighted_sum = np.sum(weighted_diff)

    return weighted_sum

# weighted_l1_capacity computes the weighted l1 norm for one vector, for capacity function
def weighted_l1_capacity(vector, c_weights):
    weighted_abs = np.abs(vector) * c_weights
    weighted_sum = np.sum(weighted_abs)
    
    return weighted_sum

# objectiveFunction computes the minimization objective function for the CASLB problem
# COPY THIS OVER FROM IMPLEMENTATIONS

# 
cpdef float singleObjective(np.ndarray x, np.ndarray cost_func, np.ndarray prev, np.ndarray w):
    return np.dot(cost_func, x) + weighted_l1_norm(x, prev, w) + weighted_l1_norm(np.zeros_like(x), x, w)

def gamma_function(gamma, U, L, D, tau, alpha):
    log = gamma * np.log( (U-L-D-(2*tau))) / ( U-(U/gamma)-D ) )
    lhs = ((U-L)/L)*log + gamma + 1 - ( (U-(2*tau*gamma)) / L)
    rhs = alpha
    return lhs - rhs

def solve_gamma(alpha, U, L, D, tau):
    guess = 1 / (1 - (2*D/U) + lambertw( ( ( (2*D/U) + (L/U) - 1 ) * math.exp(2*D/U) ) / math.e ) )
    result = newton(gamma_function, guess, args=(U, L, D, tau, alpha))
    return result


# CLIP algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
# L                         -- L
# U                         -- U
def CLIP(list vals, np.ndarray w, np.ndarray c, int dim, float L, float U, float D, float tau, np.ndarray adv, float epsilon):
    
    sol = []
    accepted = 0.0
    rob_accepted = 0.0

    adv_so_far = 0.0
    adv_accepted = 0.0

    cost_so_far = 0.0

    # get value for gamma
    gamma = solve_gamma((1+epsilon), U, L, D, tau)

    #simulate behavior of online algorithm using a for loop
    for (i, cost_func) in enumerate(vals):
        a = adv[i]
        adv_accepted += weighted_l1_capacity(a, c)
        a_prev = np.zeros(dim)
        if i != 0:
            a_prev = adv[i-1]
        adv_so_far += np.dot(cost_func, a) + weighted_l1_norm(a, a_prev, w)
        
        if accepted >= 1:
            sol.append(np.zeros(dim))
            continue
        
        remainder = (1 - accepted)
        
        if i == len(vals) - 1 and remainder > 0: # must accept last cost function
            # get the best x_T which satisfies c(x_T) = remainder
            all_bounds = [(0,1) for _ in range(0, dim)]
            
            sumConstraint = {'type': 'ineq', 'fun': lambda x: lessThanOneConstraint(x, c, accepted)}

            x0 = a
            x_T = minimize(singleObjective, x0=x0, args=(cost_func, prev, w), bounds=all_bounds, constraints=[sumConstraint]).x

            sol.append(x_T)
            accepted += weighted_l1_capacity(x_T, c)
            break

        # solve for pseudo cost-defined solution
        prev = np.zeros(dim)
        if i != 0:
            prev = sol[i-1]
        advice_t = (1+epsilon) * (adv_so_far + weighted_l1_norm(a, np.zeros(dim), w) + (1 - adv_accepted)*L)
        x_t, barx_t = clipHelper(cost_func, accepted, gamma, L, U, D, prev, w, dim, cost_so_far, advice_t, a, adv_accepted, rob_accepted)

        cost_so_far += (np.dot(cost_func,x_t) + weighted_l1_norm(x_t, prev, w))

        accepted += weighted_l1_capacity(x_t, c)
        rob_accepted += min( weighted_l1_capacity(barx_t, c), weighted_l1_capacity(x_t, c) )
        sol.append(x_t)

    cost = objectiveFunction(np.array(sol), vals, w, dim)
    return sol, cost

def consistencyConstraint(x, L, U, cost_func, prev, w, c, cost_so_far, accepted, adv_accepted, advice_t, a_t):
    c = max(0, (adv_accepted - accepted - weighted_l1_capacity(x, c)))
    compulsory = (1 - accepted - weighted_l1_capacity(x, c))*2 + c*(U-L)
    return advice_t - (cost_so_far + singleObjective(x, cost_func, prev, w) + weighted_l1_norm(x, a_t, w) + compulsory)

def lessThanOneConstraint(x, c, accepted):
    return (1-accepted) - weighted_l1_capacity(x, c)

# helper for CLIP algorithm
def clipHelper(np.ndarray cost_func, float accepted, float gamma, float L, float U, float D, np.ndarray prev, np.ndarray w, int dim, float cost_so_far, float advice_t, np.ndarray a_t, float adv_accepted, float rob_accepted):
    try:
        x0 = a_t
        all_bounds = [(0,1-accepted) for _ in range(0, dim)]

        constConstraint = {'type': 'ineq', 'fun': lambda x: consistencyConstraint(x, L, U, cost_func, prev, w, c, cost_so_far, accepted, adv_accepted, advice_t, a_t)}
        sumConstraint = {'type': 'ineq', 'fun': lambda x: lessThanOneConstraint(x, c, accepted)}

        result = minimize(clipMinimization, x0=x0, args=(cost_func, gamma, U, L, D, tau, prev, rob_accepted, w, c), method='SLSQP', bounds=all_bounds, constraints=[sumConstraint, constConstraint])
        target = result.x
        rob_target = minimize(clipMinimization, x0=x0, args=(cost_func, gamma, U, L, D, tau, prev, rob_accepted, w, c), bounds=all_bounds, constraints=sumConstraint).x
        # check if the minimization failed
        if result.success == False:
            # print("minimization failed!") 
            # this happens due to numerical instability epsilon is really small, so I just default to choose normalized a_t
            if np.sum(a_t) > 1-accepted:
                return a_t * ((1-accepted) / np.sum(a_t)), rob_target
            return a_t, rob_target
    except:
        print("something went wrong here CLIP_t= {}, z_t={}, ADV_t={}, A_t={}".format(cost_so_far, accepted, advice_t, adv_accepted))
        return a_t, np.zeros(dim)
    else:
        return target, rob_target

cpdef float thresholdFunc(float w, float U, float L, float D, float gamma):
    return U - D + (U / gamma - U + 2 * D) * np.exp( w / gamma )
# COPY THIS OVER FROM IMPLEMENTATIONS

cpdef float clipMinimization(np.ndarray x, np.ndarray cost_func, float gamma, float U, float L, float D, float tau, np.ndarray prev, float rob_accepted, np.ndarray w, np.ndarray c):
    cdef float hit_cost, pseudo_cost
    hit_cost = np.dot(cost_func, x)
    pseudo_cost = integrate.quad(thresholdFunc, rob_accepted, (rob_accepted + weighted_l1_capacity(x, c)), args=(U,L,D,tau,gamma))[0]
    return hit_cost + weighted_l1_norm(x, prev, w) - pseudo_cost