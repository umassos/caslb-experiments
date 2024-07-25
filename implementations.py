
from numba import jit
import random
import math
from scipy.optimize import linprog
import scipy.optimize
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import scipy.integrate as integrate
from scipy.special import lambertw
import ot
import numpy as np
import traceback
import cvxpy as cp
import pandas as pd
import pickle


def generate_cost_function(L, U, std, dim):
    #cost_weight = np.random.uniform(L, U, dim)
    # generate a random mean uniformly from L, U
    mean = np.random.uniform(L, U)
    # generate the cost vector from a normal distribution with mean and std
    cost_weight = np.random.normal(mean, std, dim)
    # if anything is < L or > U, set it to L or U
    for i in range(0, dim):
        if cost_weight[i] < L:
            cost_weight[i] = L
        elif cost_weight[i] > U:
            cost_weight[i] = U
    return cost_weight


# weighted_l1_norm computes the weighted L1 norm between two vectors.
# @jit
def weighted_l1_norm(vector1, vector2, phi, weights, cvxpy=False):
    if cvxpy:
        phi1 = phi @ vector1
        phi2 = phi @ vector2
        weighted_diff = cp.multiply(cp.abs(phi1 - phi2), weights)
        weighted_sum = cp.sum(weighted_diff)
    else:
        phi1 = phi @ vector1
        phi2 = phi @ vector2
        weighted_diff = np.abs(phi1 - phi2) * weights
        weighted_sum = np.sum(weighted_diff)

    return weighted_sum


# weighted l1 norm implementation for c function
# @jit
def weighted_l1_capacity(vector, weights, cvxpy=False):
    if cvxpy:
        weighted_abs = cp.multiply(cp.abs(vector), weights)
        weighted_sum = cp.sum(weighted_abs)
    else:
        weighted_abs = np.abs(vector) * weights
        weighted_sum = np.sum(weighted_abs)
    return weighted_sum


# objectiveFunctionTree computes the minimization objective function for CASLB (within the tree embedding)
# vars is the time series of decision variables (dim V x T)
# vals is the time series of cost functions (dim V x T)
# w is the weight of the switching cost 
# dim is the dimension
# @jit
def objectiveFunctionTree(vars, vals, w, c, tau, scale, dim, start_state, phi, cpy=False):
    cost = 0.0
    vars = vars.reshape((len(vals), dim))
    n = vars.shape[0]
    # n = len(vars)
    for (i, cost_func) in enumerate(vals):
        if cpy:
            cost += (cost_func @ vars[i])
        else:
            cost += np.dot(cost_func, vars[i])
        # add switching cost
        if i == 0:
            cost += weighted_l1_norm(vars[i], start_state, phi, w, cvxpy=cpy) * scale
        elif i == n-1:
            cost += weighted_l1_norm(vars[i], vars[i-1], phi, w, cvxpy=cpy) * scale
            cost += (c.T @ vars[i]) * tau
        else:
            cost += weighted_l1_norm(vars[i], vars[i-1], phi, w, cvxpy=cpy) * scale
    return cost

# objectiveFunctionSimplex computes the minimization objective for CASLB on the simplex, using the wasserstein1 distance
def objectiveFunctionSimplex(vars, gammas, vals, dist_matrix, scale, dim, c, tau, time_varying=False, cpy=False):
    cost = 0.0
    for (i, cost_func) in enumerate(vals):
        if cpy:
            cost += (cost_func @ vars[i])
        else:
            cost += np.dot(cost_func, vars[i])
    cost += (c.T @ vars[-1]) * tau
    if not time_varying:
        for gamma in gammas:
            if cpy:
                cost += cp.trace(gamma.T*dist_matrix) * scale
            else:
                cost += np.trace(gamma.T*dist_matrix) * scale
    else:
        for i, gamma in enumerate(gammas):
            if cpy:
                cost += cp.trace(gamma.T*dist_matrix[i]) * scale[i]
            else:
                cost += np.trace(gamma.T*dist_matrix[i]) * scale[i]
    return cost

# new clipObjectiveSimplex -- no optimization
def objectiveSimplexNoOpt(vars, vals, dist_matrix, scale, dim, c, tau, start_state, time_varying=False, cpy=True, debug=False):
    # get the optimal transport values using the OT library
    hitCost = 0.0
    switchCost = 0.0
    for (i, cost_func) in enumerate(vals):
        hitCost += (cost_func @ vars[i])
    for i in range(len(vals)):
        if i == 0:
            # normalize the start state and vars[i]
            start_emd = np.array(start_state) / np.sum(start_state)
            varsi_emd = np.array(vars[i]) / np.sum(vars[i])
            if time_varying:
                switchCost += ot.emd2(start_emd, varsi_emd, dist_matrix[i]) * scale[i]
            else:
                switchCost += ot.emd2(start_emd, varsi_emd, dist_matrix) * scale
        else:
            # normalize vars
            varsi_emd = np.array(vars[i]) / np.sum(vars[i])
            varsi1_emd = np.array(vars[i-1]) / np.sum(vars[i-1])
            try:
                if time_varying:
                    switchCost += ot.emd2(varsi1_emd, varsi_emd, dist_matrix[i]) * scale[i]
                else:
                    switchCost += ot.emd2(varsi1_emd, varsi_emd, dist_matrix) * scale
            except:
                print("Opt trans failed")
                print(vars[i-1])
                print(vars[i])
    switchCost += (c.T @ vars[-1]) * tau
    if not debug:
        return hitCost + switchCost
    else:
        return hitCost, switchCost

# objectiveFunctionDiscrete computes the minimization objective for CASLB 
# @jit
def objectiveFunctionDiscrete(vars, vals, dist_matrix, dim, start_state, tau, simplex_names, metric, cpy=False):
    cost = 0.0
    n = vars.shape[0]
    for (i, cost_func) in enumerate(vals):
        if cpy:
            cost += (cost_func @ vars[i])
        else:
            cost += np.dot(cost_func, vars[i])
        # add switching cost
        # temporal - take the L1 norm of the difference between the two and multiply by tau
        if i == 0:
            cost += cp.norm(vars[i] - start_state, 1) * tau
        elif i == n-1:
            cost += cp.norm(vars[i] - vars[i-1], 1) * tau
            cost += cp.norm(start_state - vars[i], 1) * tau
        else:
            cost += cp.norm(vars[i] - vars[i-1], 1) * tau
        # spatial - if the location has changed, charge based on the distance
        # get the first non-zero index of both vectors
        if (vars[i].T @ vars[i]) != 0.5:
            # get the index of the first non-zero element
            loc1 = np.where(vars[i] == 1)[0][0]
            loc2 = np.where(vars[i-1] == 1)[0][0]
            if loc1 != loc2:
                cost += dist_matrix[simplex_names[loc1], simplex_names[loc2]]
    return cost
    

# @jit
def negativeObjectiveSimplex(vars, gammas, vals, dist_matrix, scale, dim, c, tau, cpy=False):
    return -1 * objectiveFunctionSimplex(vars, gammas, vals, dist_matrix, scale, dim, c, tau, cpy=cpy)


# computing the optimal solution
# @jitdef optimalSolution(simplex_cost_functions, dist_matrix, tau, scale, c_simplex, d, start_state):
def optimalSolution(simplex_cost_functions, dist_matrix, tau, scale, c_simplex, d, start_state, time_varying=False, alt_cost_functions=None):
    T = len(simplex_cost_functions)
    # declare variables
    x = cp.Variable((T, d))
    gammas = [cp.Variable((d,d)) for _ in range(0, T)]
    constraints = [0 <= x, x <= 1]
    # add deadline constraint
    c = np.array(c_simplex)
    constraints += [cp.sum(x @ c.T) == 1]
    # # add location constraint
    # A = -1*np.ones((int(d/2), d))
    # for j in range(0, d):
    #     row = j//2
    #     A[row][j] = 1
    # for i in range(0, T):
    #     constraints += [A @ x[i] == 1]
    # add Wasserstein constraints
    for i in range(T):
        constraints += [cp.sum(x[i]) == 1]
    for i in range(0, T):
        constraints += [gammas[i] >= 0]
        # each x[i] should sum to 1
        if i == 0:
            constraints += [gammas[i] @ np.ones(d) == start_state]
            constraints += [gammas[i].T @ np.ones(d) == x[i]]
        else:
            constraints += [gammas[i] @ np.ones(d) == x[i-1]]
            constraints += [gammas[i].T @ np.ones(d) == x[i]] 
    prob = cp.Problem(cp.Minimize(objectiveFunctionSimplex(x, gammas, simplex_cost_functions, dist_matrix, scale, d, c_simplex, tau, time_varying=time_varying, cpy=True)), constraints)
    prob.solve(solver=cp.ECOS_BB)
    # print("status:", prob.status)
    # print("optimal value", prob.value)
    # print("optimal var", x.value)
    if prob.status == 'optimal' or prob.status == 'optimal_inaccurate':
        if alt_cost_functions is not None:
            return x.value, gammas, objectiveSimplexNoOpt(x.value, alt_cost_functions, dist_matrix, scale, d, c_simplex, tau, start_state, time_varying=time_varying, cpy=True)
        return x.value, objectiveSimplexNoOpt(x.value, simplex_cost_functions, dist_matrix, scale, d, c_simplex, tau, start_state, time_varying=time_varying, cpy=False)
    else:
        return x.value, 0.0


# computing the adversarial solution using cvxpy instead
# @jit
def adversarialSolution(simplex_cost_functions, dist_matrix, tau, scale, c_simplex, d, start_state):
    T = len(simplex_cost_functions)
    # declare variables
    x = cp.Variable((T, d))
    gammas = [cp.Variable((d,d)) for _ in range(0, T)]
    constraints = [0 <= x, x <= 1]
    # add deadline constraint
    c = np.array(c_simplex)
    constraints += [cp.sum(x @ c.T) == 1]
    for i in range(T):
        constraints += [cp.sum(x[i]) == 1]
    for i in range(0, T):
        constraints += [gammas[i] >= 0]
        # each x[i] should sum to 1
        if i == 0:
            constraints += [gammas[i] @ np.ones(d) == start_state]
            constraints += [gammas[i].T @ np.ones(d) == x[i]]
        else:
            constraints += [gammas[i] @ np.ones(d) == x[i-1]]
            constraints += [gammas[i].T @ np.ones(d) == x[i]] 
    prob = cp.Problem(cp.Minimize(negativeObjectiveSimplex(x, gammas, simplex_cost_functions, dist_matrix, scale, d, c_simplex, tau, cpy=True)), constraints)
    prob.solve(solver=cp.ECOS_BB)
    # print("status:", prob.status)
    # print("optimal value", prob.value)
    # print("optimal var", x.value)
    if prob.status == 'optimal' or prob.status == 'optimal_inaccurate':
        return x.value, objectiveSimplexNoOpt(x.value, simplex_cost_functions, dist_matrix, scale, d, c_simplex, tau, start_state, cpy=False)
    else:
        return x.value, 0.0

# def singleObjective(x, cost_func, previous, w, scale):
#     return np.dot(cost_func, x) + weighted_l1_norm(x, previous, w) + weighted_l1_norm(np.zeros_like(x), x, w) * scale
# @jit
def singleObjective(x, cost_func, previous, phi, w, scale, start, cpy=True):
    return (cost_func @ x) + weighted_l1_norm(x, previous, phi, w, cvxpy=cpy) * scale + weighted_l1_norm(start, x, phi, w, cvxpy=cpy) * scale

# PCM algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
# L                         -- L
# U                         -- U
# D                         -- diameter of the metric
# @jit
def PCM(vals, w, scale_list, c, job_length, phi_list, dim, L, U, D, tau, start, time_varying=False):
    sol = []
    accepted = 0.0

    # get value for eta
    eta = 1 / ( (U-D)/U + lambertw( ( (D+L-U+(2*tau)) * math.exp((D-U)/U) )/U ) )

    #simulate behavior of online algorithm using a for loop
    for (i, cost_func) in enumerate(vals):
        phi = phi_list
        scale = scale_list
        if time_varying:
            phi = phi_list[i]
            scale = scale_list[i]

        if accepted >= 1:
            # check the previous solution
            previous = sol[i-1].copy()
            # if the c function is non-zero, then we need to move everything to OFF states
            if (c.T @ previous) != 0:
                next_x = np.zeros(dim)
                for j in range(dim):
                    if c[j] > 0:
                        next_x[j+1] += previous[j]
                    else:
                        next_x[j] += previous[j]
                sol.append(next_x)
            else:
                sol.append(previous)
            continue
        
        remainder = (1 - accepted)
        remaining_length = job_length - (remainder * job_length)
        # if remaining length is 2.5, need to compulsory trade on last three steps

        if i >= len(vals) - np.ceil(remaining_length) and remainder > 0: # must accept last cost functions
            # get the best x_T which satisfies c(x_T) = remainder
            # use cvxpy
            x = cp.Variable(dim)
            target_capacity = min(1/job_length, remainder)
            constraints = [0 <= x, x <= 1, cp.sum(x) == 1, c.T @ x == target_capacity]
            # Wasserstein constraints
            # x, gamma_ot, gamma_ot_2, dist_matrix, cost_func, scale,
            prob = cp.Problem(cp.Minimize(singleObjective(x, cost_func, previous, phi, w, scale, start)), constraints)
            prob.solve(solver=cp.CLARABEL)
            x_T = x.value
            sol.append(x_T)
            accepted += c.T @ x_T
            continue

        # solve for pseudo cost-defined solution
        previous = np.zeros(dim)
        if i != 0:
            previous = sol[i-1]
        x_t = pcmHelper(cost_func, accepted, eta.real, L, U, D, tau, previous, w, scale, c, phi, dim, start)
        # normalize x_t
        # x_t = x_t / np.sum(x_t)

        # print(np.dot(cost_func,x_t) + weighted_l1_norm(x_t, previous, w))
        # print(integrate.quad(thresholdFunc, 0, (0 + np.linalg.norm(x_t, ord=1)), args=(U,L,D,eta))[0])

        accepted += c.T @ x_t
        sol.append(x_t) 

    cost = objectiveFunctionTree(np.array(sol), vals, w, c, tau, scale, dim, start, phi)
    return sol, cost

# helper for PCM algorithm
# @jit
def pcmHelper(cost_func, accepted, eta, L, U, D, tau, previous, w, scale, c, phi, dim, start):
    # use cvxpy to solve the problem
    x = cp.Variable(dim, pos=True)
    constraints = [0 <= x, x <= 1, cp.sum(x) == 1, c.T @ x <= (1-accepted)]
    prob = cp.Problem(cp.Minimize(pcmMinimization(x, cost_func, eta, U, L, D, tau, previous, accepted, w, phi, scale, c)), constraints)
    prob.solve(solver=cp.CLARABEL)
    target = x.value
    if np.sum(target) > 1.01:
        print("sum of target is greater than 1!  sum: {}".format(sum(target)))
    return target

def thresholdFunc( w,  U,  L,  D,  tau,  eta):
    return U - tau + (U / eta - U + D + tau) * np.exp( w / eta )

def thresholdAntiDeriv(w, U, L, D, tau, eta):
    return U*w - tau*w + (tau * eta - U * eta + D*eta + U) * cp.exp( w / eta )

def pcmMinimization(x, cost_func, eta, U, L, D, tau, previous, accepted, w, phi, scale, c):
    hit_cost = (cost_func @ x)
    next_accepted = (accepted + (c.T @ x))
    pseudo_cost_a = thresholdAntiDeriv(accepted, U,L,D,tau,eta)
    pseudo_cost_b = thresholdAntiDeriv(next_accepted, U,L,D,tau,eta)
    #pseudo_cost = integrate.quad(thresholdFunc, accepted, (accepted + (c.T @ x)), args=(U,L,D,tau,eta))[0]
    return hit_cost + (weighted_l1_norm(x, previous, phi, w, cvxpy=True) * scale) - (pseudo_cost_b - pseudo_cost_a) #pseudo_cost


# "greedy" algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
def greedy(vals, w, scale, c, job_length, dim, tau, dist_matrix, start, time_varying=False):
    # choose a the minimum dimension at time 0 and ramp up there
    min_dim = np.argmin(vals[0][np.nonzero(vals[0])]) * 2
    remaining = 1.0

    # construct a solution which ramps up to 1 on the selected dimension at the first time step
    sol = []
    for i, _ in enumerate(vals):
        prev = start
        if i != 0:
            prev = sol[i-1]
        if remaining <= 0:
            # get the closest off state
            if c.T @ sol[-1] != 0:
                next_x = np.zeros(dim)
                for j in range(dim):
                    if c[j] > 0:
                        next_x[j+1] += prev[j]
                    else:
                        next_x[j] += prev[j]
                sol.append(next_x)
            else:
                sol.append(sol[-1])
            continue
        x_t = np.zeros(dim)
        x_t[min_dim] = 1.0
        sol.append(x_t)
        remaining = remaining - (c.T @ x_t)
    
    #objectiveSimplexNoOpt(vars, vals, dist_matrix, scale, dim, c, tau, start_state, cpy=True):

    cost = objectiveSimplexNoOpt(np.array(sol), vals, dist_matrix, scale, dim, c, tau, start, time_varying=time_varying, cpy=False)
    return sol, cost

# "agnostic" algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
def agnostic(vals, w, scale, c, job_length, dim, tau, dist_matrix, start, time_varying=False):
    remaining = 1.0
    cost = 0.0
    
    # choose the current ON state and turn ON until the job is done
    on_state = np.zeros(dim)
    for j in range(dim):
        if c[j] > 0:
            on_state[j] += start[j+1]

    # construct a solution which ramps up to on state at original location
    sol = []
    for i, val in enumerate(vals):
        prev = start
        if i != 0:
            prev = sol[-1]
        if remaining <= 0:
            # get the closest off state
            if c.T @ sol[-1] != 0:
                next_x = np.zeros(dim)
                for j in range(dim):
                    if c[j] > 0:
                        next_x[j+1] += prev[j]
                    else:
                        next_x[j] += prev[j]
                sol.append(next_x)
            else:
                sol.append(sol[-1])
            continue
        sol.append(on_state)
        remaining = remaining - (c.T @ on_state)

    cost = objectiveSimplexNoOpt(np.array(sol), vals, dist_matrix, scale, dim, c, tau, start, time_varying=time_varying, cpy=False)
    return sol, cost

# "delayedGreedy" algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
def delayedGreedy(vals, adv_vals, w, scale, c, job_length, dim, tau, dist_matrix, start, time_varying=False):
    # choose the minimum dimension across
    start_time = 0
    min_dim = 0
    best_val_so_far = np.inf
    for i in range(len(vals)):
        min_dim_cur = np.argmin(adv_vals[i][np.nonzero(adv_vals[i])]) * 2
        value_cur = adv_vals[i][min_dim_cur]
        if value_cur < best_val_so_far:
            best_val_so_far = value_cur
            min_dim = min_dim_cur
            start_time = i
    if start_time > len(vals) - job_length:
        start_time = len(vals) - job_length
    remaining = 1.0

    # construct a solution which ramps up to 1 on the selected dimension, starting at the start_time step
    sol = []
    for i, _ in enumerate(vals):
        prev = start
        if i != 0:
            prev = sol[i-1]
        if remaining <= 0:
            # get the closest off state
            if c.T @ sol[-1] != 0:
                next_x = np.zeros(dim)
                for j in range(dim):
                    if c[j] > 0:
                        next_x[j+1] += prev[j]
                    else:
                        next_x[j] += prev[j]
                sol.append(next_x)
            else:
                sol.append(sol[-1])
            continue
        if i >= start_time:
            x_t = np.zeros(dim)
            x_t[min_dim] = 1.0
            sol.append(x_t)
            remaining = remaining - (c.T @ x_t)
        else:
            x_t = start
            sol.append(x_t)
    
    cost = objectiveSimplexNoOpt(np.array(sol), vals, dist_matrix, scale, dim, c, tau, start, time_varying=time_varying, cpy=False)
    return sol, cost

# "simple threshold" algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
# L                         -- L
# U                         -- U
def threshold(vals, w, scale, c, job_length, phi, dim, L, U, D, tau, start, dist_matrix, time_varying=False):
    threshold = np.sqrt(L*U)

    remaining = 1.0

    # construct a solution which ramps up to 1 at the dimension <= root UL 
    sol = []
    for i, cost_func in enumerate(vals):
        prev = start
        if i != 0:
            prev = sol[i-1]
        if remaining <= 0:
            # get the closest off state
            if c.T @ sol[-1] != 0:
                next_x = np.zeros(dim)
                for j in range(dim):
                    if c[j] > 0:
                        next_x[j+1] += prev[j]
                    else:
                        next_x[j] += prev[j]
                sol.append(next_x)
            else:
                sol.append(sol[-1])
            continue
        if i == len(vals) - np.ceil(remaining*job_length): # must accept last cost function
            # get the closest on state
            next_x = np.zeros(dim)
            for j in range(dim):
                if c[j] > 0:
                    next_x[j] += prev[j]
                    next_x[j] += prev[j+1]
            sol.append(next_x)
            continue

        thresholding = cost_func[np.nonzero(cost_func)] <= threshold
        # in the first location where thresholding is true, we can set the value to 1
        x_t = start
        for j in range(0, len(thresholding)):
            if thresholding[j]:
                x_t = np.zeros(dim)
                x_t[j*2] = 1.0
                break
        
        sol.append(x_t)
        remaining = remaining - (c.T @ x_t)

    cost = objectiveSimplexNoOpt(np.array(sol), vals, dist_matrix, scale, dim, c, tau, start, time_varying=time_varying, cpy=False)
    return sol, cost