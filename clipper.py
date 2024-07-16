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
import cvxpy as cp
import pandas as pd
import pickle

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
def objectiveFunctionTree(vars, vals, w, scale, dim, start_state, phi, cpy=False):
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
            cost += weighted_l1_norm(start_state, vars[i], phi, w, cvxpy=cpy) * scale
        else:
            cost += weighted_l1_norm(vars[i], vars[i-1], phi, w, cvxpy=cpy) * scale
    return cost


# objectiveFunctionSimplex computes the minimization objective for CASLB on the simplex, using the wasserstein1 distance
# @jit
def objectiveFunctionSimplex(vars, gammas, vals, dist_matrix, scale, dim, start_state, cpy=False):
    cost = 0.0
    n = vars.shape[0]
    for (i, cost_func) in enumerate(vals):
        if cpy:
            cost += (cost_func @ vars[i])
        else:
            cost += np.dot(cost_func, vars[i])
    for gamma in gammas:
        cost += cp.trace(gamma.T*dist_matrix) * scale
    return cost

# @jit
def clipObjectiveSimplex(vars, vals, dist_matrix, scale, dim, start_state, cpy=True):
    # solve for the optimal transport matrices
    T = len(vals)
    gammas = [cp.Variable((dim,dim)) for _ in range(0, T+1)]
    constraints = []
    # Wasserstein constraints
    for i in range(0, T+1):
        constraints += [gammas[i] >= 0]
        # each x[i] should sum to 1
        if i == 0:
            constraints += [gammas[i] @ np.ones(dim) == start_state]
        else:
            constraints += [gammas[i] @ np.ones(dim) == vars[i-1]]
        if i == T:
            constraints += [gammas[i].T @ np.ones(dim) == start_state]
        else:
            constraints += [gammas[i].T @ np.ones(dim) == vars[i]]
    prob = cp.Problem(cp.Minimize(objectiveFunctionSimplex(vars, gammas, vals, dist_matrix, scale, dim, start_state, cpy=True)), constraints)
    prob.solve(solver=cp.ECOS)
    return prob.value

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


# solve for gamma using scipy or whatever
def gamma_function(gamma, U, L, D, tau, alpha):
    log = gamma * np.log( (U-L-D-(2*tau)) / ( U-(U/gamma)-D ) )
    lhs = ((U-L)/L)*log + gamma + 1 - ( (U-(2*tau*gamma)) / L)
    rhs = alpha
    return lhs - rhs

def solve_gamma(alpha, U, L, D, tau):
    guess = 1 / (1 - (2*D/U) + lambertw( ( ( (2*D/U) + (L/U) - 1 ) * math.exp(2*D/U) ) / math.e ) )
    result = newton(gamma_function, guess, args=(U, L, D, tau, alpha))
    return result


def singleObjective(x, gamma_ot, c, tau, dist_matrix, cost_func, scale, cpy=True):
    return (cost_func @ x) + cp.trace(gamma_ot*dist_matrix) * scale + (c.T @ x)*tau


# CLIP algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
# L                         -- L
# U                         -- U
def Clipper(vals, w, scale, c, phi, dim, L, U, D, tau, adv, adv_gamma_ots, dist_matrix, epsilon, start):
    sol = []
    accepted = 0.0
    rob_accepted = 0.0

    adv_so_far = 0.0
    adv_accepted = 0.0

    cost_so_far = 0.0

    # get value for gamma
    gamma = solve_gamma((1+epsilon), U, L, D, tau)
    # only get the real part of the gamma
    gamma = gamma.real

    #simulate behavior of online algorithm using a for loop
    for (i, cost_func) in enumerate(vals):
        a = adv[i]
        a_gamma_ot = adv_gamma_ots[i]
        adv_accepted += c.T @ a
        adv_so_far += (cost_func @ a) + np.trace(a_gamma_ot.T*dist_matrix) * scale
        if i == len(a)-1:
            a_gamma_ot = adv_gamma_ots[i+1]
            adv_so_far += np.trace(a_gamma_ot.T*dist_matrix) * scale
        
        if accepted >= 1:
            # check the previous solution
            previous = sol[i-1]
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
        
        if i == len(vals) - 1 and remainder > 0: # must accept last cost function
            # get the best x_T which satisfies c(x_T) = remainder
            # use cvxpy
            x = cp.Variable(dim)
            gamma_ot = cp.Variable((dim, dim))
            constraints = [0 <= x, x <= 1, cp.sum(x) == 1, c.T @ x == remainder]
            # Wasserstein constraints
            constraints += [gamma_ot >= 0]
            constraints += [gamma_ot @ np.ones(dim) == sol[-1], gamma_ot.T @ np.ones(dim) == x]
            # x, gamma_ot, gamma_ot_2, dist_matrix, cost_func, scale,
            prob = cp.Problem(cp.Minimize(singleObjective(x, gamma_ot, c, tau, dist_matrix, cost_func, scale, cpy=True)), constraints)
            prob.solve(solver='ECOS')
            x_T = x.value
            sol.append(x_T)
            accepted += c.T @ x_T
            break

        # solve for pseudo cost-defined solution
        prev = start
        if i != 0:
            prev = sol[i-1]
        advice_t = (1+epsilon) * (adv_so_far + (c.T @ a)*tau + (1 - adv_accepted)*L)
        x_t, gamma_ot, barx_t = clipHelper(cost_func, accepted, gamma, L, U, D, tau, prev, w, dist_matrix, scale, c, phi, dim, cost_so_far, advice_t, a, adv_accepted, rob_accepted)

        cost_so_far += (cost_func @ x_t) + np.trace(gamma_ot.T*dist_matrix) * scale

        accepted += c.T @ x_t
        rob_accepted += min( (c.T @ barx_t), (c.T @ x_t) )
        sol.append(x_t)

    cost = clipObjectiveSimplex(np.array(sol), vals, dist_matrix, scale, dim, start)
    return sol, cost

def consistencyConstraint(x, gamma_ot, gamma_ot_opt, dist_matrix, L, U, cost_func, prev, w, c, tau, cost_so_far, accepted, adv_accepted, scale):
    comp = cp.max(cp.vstack([0, (adv_accepted - accepted - c.T @ x)]))
    compulsory = (1 - accepted - (c.T @ x))*2 + comp*(U-L)
    return cost_so_far + singleObjective(x, gamma_ot, c, tau, dist_matrix, cost_func, scale, cpy=True) + cp.trace(gamma_ot_opt*dist_matrix) * scale + compulsory

# helper for CLIP algorithm
def clipHelper(cost_func, accepted, gamma, L, U, D, tau, prev, w, dist_matrix, scale, c, phi, dim, cost_so_far, advice_t, a_t, adv_accepted, rob_accepted):
    # try:
    #     x0 = a_t
    #     all_bounds = [(0,1-accepted) for _ in range(0, dim)]

    #     constConstraint = {'type': 'ineq', 'fun': lambda x: consistencyConstraint(x, L, U, cost_func, prev, w, c, cost_so_far, accepted, adv_accepted, advice_t, a_t)}

    #     result = minimize(clipperMinimization, x0=x0, args=(cost_func, gamma, U, L, D, tau, prev, rob_accepted, w, phi, scale, c), method='SLSQP', bounds=all_bounds, constraints=[sumConstraint, constConstraint])
    #     target = result.x
    #     rob_target = minimize(clipperMinimization, x0=x0, args=(cost_func, gamma, U, L, D, tau, prev, rob_accepted, w, phi, scale, c), bounds=all_bounds, constraints=sumConstraint).x
    #     # check if the minimization failed
    #     if result.success == False:
    #         # print("minimization failed!") 
    #         # this happens due to numerical instability epsilon is really small, so I just default to choose normalized a_t
    #         if np.sum(a_t) > 1-accepted:
    #             return a_t * ((1-accepted) / np.sum(a_t)), rob_target
    #         return a_t, rob_target
    # except:
    #     print("something went wrong here CLIP_t= {}, z_t={}, ADV_t={}, A_t={}".format(cost_so_far, accepted, advice_t, adv_accepted))
    #     # return a_t, np.zeros(dim)
    # else:
    #     print('blah')
    #     # return target, rob_target
    
    # use cvxpy to solve the problem
    x = cp.Variable(dim, pos=True)
    gamma_ot = cp.Variable((dim, dim))
    gamma_ot_opt = cp.Variable((dim, dim))
    x_bar = cp.Variable(dim, pos=True)
    adv_constraints = [0 <= x, x <= 1, cp.sum(x) == 1, c.T @ x <= (1-accepted)]
    # add Wasserstein constraints
    adv_constraints += [gamma_ot >= 0, gamma_ot_opt >= 0]
    adv_constraints += [gamma_ot @ np.ones(dim) == prev, gamma_ot.T @ np.ones(dim) == x]
    adv_constraints += [gamma_ot_opt @ np.ones(dim) == x, gamma_ot_opt.T @ np.ones(dim) == a_t]
    # add consistency constraint
    adv_constraints += [consistencyConstraint(x, gamma_ot, gamma_ot_opt, dist_matrix, L, U, cost_func, prev, w, c, tau, cost_so_far, accepted, adv_accepted, scale) <= advice_t]
    adv_prob = cp.Problem(cp.Minimize(clipperMinimization(x, cost_func, gamma, U, L, D, tau, prev, accepted, w, phi, scale, c)), adv_constraints)
    adv_prob.solve(solver=cp.CLARABEL)
    target = x.value

    robust_constraints = [0 <= x, x <= 1, cp.sum(x) == 1, c.T @ x <= (1-accepted)]
    rob_prob = cp.Problem(cp.Minimize(clipperMinimization(x_bar, cost_func, gamma, U, L, D, tau, prev, accepted, w, phi, scale, c)), robust_constraints)
    rob_prob.solve(solver=cp.CLARABEL)
    rob_target = x_bar.value

    if adv_prob.status == 'optimal':
        return target, gamma_ot.value, rob_target
    else:
        return a_t, gamma_ot.value, rob_target

def thresholdFunc( w,  U,  L,  D,  tau,  gamma):
    return U - tau + (U / gamma - U + D + tau) * np.exp( w / gamma )

def thresholdAntiDeriv(w, U, L, D, tau, gamma):
    return U*w - tau*w + (tau * gamma - U * gamma + D*gamma + U) * cp.exp( w / gamma )

def clipperMinimization(x, cost_func, gamma, U, L, D, tau, prev, rob_accepted, w, phi, scale, c):
    hit_cost = (cost_func @ x)
    next_accepted = (rob_accepted + (c.T @ x))
    pseudo_cost_a = thresholdAntiDeriv(rob_accepted, U,L,D,tau,gamma)
    pseudo_cost_b = thresholdAntiDeriv(next_accepted, U,L,D,tau,gamma)
    #pseudo_cost = integrate.quad(thresholdFunc, accepted, (accepted + (c.T @ x)), args=(U,L,D,tau,eta))[0]
    return hit_cost + (weighted_l1_norm(x, prev, phi, w, cvxpy=True) * scale) - (pseudo_cost_b - pseudo_cost_a)#pseudo_cost