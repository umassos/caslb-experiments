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

# objectiveFunction computes the minimization objective function for the OCS problem.
# vars is the time series of decision variables (dim d x T)
# vals is the time series of cost functions (dim d x T)
# w is the weight of the switching cost 
# dim is the dimension
def objectiveFunction(vars, vals, w, dim):
    cost = 0.0
    vars = vars.reshape((len(vals), dim))
    n = vars.shape[0]
    # n = len(vars)
    for (i, cost_func) in enumerate(vals):
        cost += np.dot(cost_func, vars[i])
        # add switching cost
        if i == 0:
            cost += weighted_l1_norm(vars[i], np.zeros(dim), w)
        elif i == n-1:
            cost += weighted_l1_norm(vars[i], vars[i-1], w)
            cost += weighted_l1_norm(np.zeros(dim), vars[i], w)
        else:
            cost += weighted_l1_norm(vars[i], vars[i-1], w)
    return cost

# 
cpdef float singleObjective(np.ndarray x, np.ndarray cost_func, np.ndarray prev, np.ndarray w):
    return np.dot(cost_func, x) + weighted_l1_norm(x, prev, w) + weighted_l1_norm(np.zeros_like(x), x, w)

def gamma_function(gamma, U, L, beta, eta):
    log = gamma * np.log( (L + 2*beta - U) / ( (U/gamma) + 2*beta - U) )
    lhs = ((U-L)/L)*log + gamma + 1 - (U/L)
    rhs = eta
    return lhs - rhs

def solve_gamma(eta, U, L, beta):
    guess = 1 / (1 - (2*beta/U) + lambertw( ( ( (2*beta/U) + (L/U) - 1 ) * math.exp(2*beta/U) ) / math.e ) )
    result = newton(gamma_function, guess, args=(U, L, beta, eta))
    return result


# CLIP algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
# L                         -- L
# U                         -- U
def CLIP(list vals, np.ndarray w, int dim, float L, float U, np.ndarray adv, float epsilon):
    
    sol = []
    accepted = 0.0
    rob_accepted = 0.0

    adv_so_far = 0.0
    adv_accepted = 0.0

    cost_so_far = 0.0

    # get value for beta
    beta = np.max(w)

    # get value for gamma
    gamma = solve_gamma((1+epsilon), U, L, beta)

    #simulate behavior of online algorithm using a for loop
    for (i, cost_func) in enumerate(vals):
        a = adv[i]
        adv_accepted += np.linalg.norm(a, ord=1)
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
            
            sumConstraint = {'type': 'ineq', 'fun': lambda x: lessThanOneConstraint(x, accepted)}

            x0 = a
            x_T = minimize(singleObjective, x0=x0, args=(cost_func, prev, w), bounds=all_bounds, constraints=[sumConstraint]).x

            sol.append(x_T)
            accepted += np.linalg.norm(x_T, ord=1)
            break

        # solve for pseudo cost-defined solution
        prev = np.zeros(dim)
        if i != 0:
            prev = sol[i-1]
        advice_t = (1+epsilon) * (adv_so_far + weighted_l1_norm(a, np.zeros(dim), w) + (1 - adv_accepted)*L)
        x_t, barx_t = clipHelper(cost_func, accepted, gamma, L, U, beta, prev, w, dim, cost_so_far, advice_t, a, adv_accepted, rob_accepted)

        cost_so_far += (np.dot(cost_func,x_t) + weighted_l1_norm(x_t, prev, w))

        accepted += np.linalg.norm(x_t, ord=1)
        rob_accepted += min( np.linalg.norm(barx_t, ord=1), np.linalg.norm(x_t, ord=1) )
        sol.append(x_t)

    cost = objectiveFunction(np.array(sol), vals, w, dim)
    return sol, cost

def consistencyConstraint(x, L, U, cost_func, prev, w, cost_so_far, accepted, adv_accepted, advice_t, a_t):
    c = max(0, (adv_accepted - accepted - np.linalg.norm(x, ord=1)))
    compulsory = (1 - accepted - np.linalg.norm(x, ord=1))*2 + c*(U-L)
    return advice_t - (cost_so_far + singleObjective(x, cost_func, prev, w) + weighted_l1_norm(x, a_t, w) + compulsory)

def lessThanOneConstraint(x, accepted):
    return (1-accepted) - np.sum(x)

# helper for CLIP algorithm
def clipHelper(np.ndarray cost_func, float accepted, float gamma, float L, float U, float beta, np.ndarray prev, np.ndarray w, int dim, float cost_so_far, float advice_t, np.ndarray a_t, float adv_accepted, float rob_accepted):
    try:
        x0 = a_t
        all_bounds = [(0,1-accepted) for _ in range(0, dim)]

        constConstraint = {'type': 'ineq', 'fun': lambda x: consistencyConstraint(x, L, U, cost_func, prev, w, cost_so_far, accepted, adv_accepted, advice_t, a_t)}
        sumConstraint = {'type': 'ineq', 'fun': lambda x: lessThanOneConstraint(x, accepted)}

        result = minimize(clipMinimization, x0=x0, args=(cost_func, gamma, U, L, beta, prev, rob_accepted, w), method='SLSQP', bounds=all_bounds, constraints=[sumConstraint, constConstraint])
        target = result.x
        rob_target = minimize(clipMinimization, x0=x0, args=(cost_func, gamma, U, L, beta, prev, rob_accepted, w), bounds=all_bounds, constraints=sumConstraint).x
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

cpdef float thresholdFunc(float w, float U, float L, float beta, float gamma):
    return U - beta + (U / gamma - U + 2 * beta) * np.exp( w / gamma )

cpdef float clipMinimization(np.ndarray x, np.ndarray cost_func, float gamma, float U, float L, float beta, np.ndarray prev, float rob_accepted, np.ndarray w):
    cdef float hit_cost, pseudo_cost
    hit_cost = np.dot(cost_func, x)
    pseudo_cost = integrate.quad(thresholdFunc, rob_accepted, (rob_accepted + np.linalg.norm(x, ord=1)), args=(U,L,beta,gamma))[0]
    return hit_cost + weighted_l1_norm(x, prev, w) - pseudo_cost