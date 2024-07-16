
from numba import jit
import random
import math
from scipy.optimize import linprog
import scipy.optimize
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import scipy.integrate as integrate
from scipy.special import lambertw
# import ot
import numpy as np
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
def negativeObjectiveSimplex(vars, vals, dist_matrix, dim, start_state, tau, simplex_names, metric, cpy=False):
    return -1 * objectiveFunctionSimplex(vars, vals, dist_matrix, dim, start_state, tau, simplex_names, metric, cpy=cpy)



# computing the optimal solution
# @jit
def optimalSolution(simplex_cost_functions, dist_matrix, scale, c_simplex, d, start_state, alt_cost_functions=None):
    T = len(simplex_cost_functions)
    # declare variables
    x = cp.Variable((T, d))
    gammas = [cp.Variable((d,d)) for _ in range(0, T+1)]
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
    for i in range(0, T+1):
        constraints += [gammas[i] >= 0]
        # each x[i] should sum to 1
        if i == 0:
            constraints += [gammas[i] @ np.ones(d) == start_state]
        else:
            constraints += [gammas[i] @ np.ones(d) == x[i-1]]
        if i == T:
            constraints += [gammas[i].T @ np.ones(d) == start_state]
        else:
            constraints += [gammas[i].T @ np.ones(d) == x[i]]
    prob = cp.Problem(cp.Minimize(objectiveFunctionSimplex(x, gammas, simplex_cost_functions, dist_matrix, scale, d, start_state, cpy=True)), constraints)
    prob.solve(solver=cp.ECOS)
    # print("status:", prob.status)
    # print("optimal value", prob.value)
    # print("optimal var", x.value)
    if prob.status == 'optimal':
        if alt_cost_functions is not None:
            # valueGammas = []
            # for i in range(0, T+1):
            #     valueGammas.append(gammas[i].value)
            return x.value, gammas, objectiveFunctionSimplex(x, gammas, alt_cost_functions, dist_matrix, scale, d, start_state, cpy=True).value
        return x.value, prob.value
    else:
        return x.value, 0.0


# computing the adversarial solution using cvxpy instead
# @jit
def adversarialSolution(simplex_cost_functions, dist_matrix, scale, c_simplex, d, start_state):
    T = len(simplex_cost_functions)
    # declare variables
    x = cp.Variable((T, d))
    gammas = [cp.Variable((d,d)) for _ in range(0, T+1)]
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
    for i in range(0, T+1):
        constraints += [gammas[i] >= 0]
        # each x[i] should sum to 1
        if i == 0:
            constraints += [gammas[i] @ np.ones(d) == start_state]
        else:
            constraints += [gammas[i] @ np.ones(d) == x[i-1]]
        if i == T:
            constraints += [gammas[i].T @ np.ones(d) == start_state]
        else:
            constraints += [gammas[i].T @ np.ones(d) == x[i]]
    prob = cp.Problem(cp.Minimize(negativeObjectiveSimplex(x, gammas, simplex_cost_functions, dist_matrix, scale, d, start_state, cpy=True)), constraints)
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", x.value)
    if prob.status == 'optimal':
        return x.value, prob.value
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
def PCM(vals, w, scale, c, phi, dim, L, U, D, tau, start):
    sol = []
    accepted = 0.0

    # get value for eta
    eta = 1 / ( (U-D)/U + lambertw( ( (D+L-U+(2*tau*scale)) * math.exp((D-U)/U) )/U ) )

    #simulate behavior of online algorithm using a for loop
    for (i, cost_func) in enumerate(vals):
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
        
        if i == len(vals) - 1: # must accept last cost function
            # get the best x_T which satisfies c(x_T) = remainder
            # use cvxpy
            x = cp.Variable(dim)
            constraints = [0 <= x, x <= 1, cp.sum(x) == 1, c.T @ x == remainder]
            prob = cp.Problem(cp.Minimize(singleObjective(x, cost_func, previous, phi, w, scale, start)), constraints)
            prob.solve(solver='ECOS')
            x_T = x.value
            sol.append(x_T)
            accepted += c.T @ x_T
            break

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

    cost = objectiveFunctionTree(np.array(sol), vals, w, scale, dim, start, phi)
    return sol, cost

# helper for PCM algorithm
# @jit
def pcmHelper(cost_func, accepted, eta, L, U, D, tau, previous, w, scale, c, phi, dim, start):
    # cdef np.ndarray target, x0, A
    # cdef list all_bounds, b
    # try:
    #     # initialize x0 to put probability mass at the minimum hitting cost
    #     index = np.argmin(cost_func[np.nonzero(cost_func)])
    #     x0 = np.zeros(dim)
    #     x0[index] = 1.0
    #     # x0 = start
    #     all_bounds = [(0,1) for _ in range(0, dim)]
    #     A = np.ones(dim)
    #     b = [1]
    #     constraint = LinearConstraint(A, lb=b)
    #     # sumConstraint = {'type': 'eq', 'fun': lambda x: exactConstraint(x)}
    #     target = minimize(pcmMinimization, x0=x0, args=(cost_func, eta, U, L, D, tau, previous, accepted, w, phi, scale, c), bounds=all_bounds, constraints=constraint, method='COBYQA').x
    # except:
    #     print("something went wrong here w_j={}".format(accepted))
    #     # what went wrong
    #     return start
    # else:
    #     return target
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

# cpdef float thresholdAntiDeriv(float w, float U, float L, float D, float tau, float eta):
#     return U*w - tau*w + (tau * eta - U * eta + D*eta + U) * np.exp( w / eta )

# cpdef float pcmMinimization(np.ndarray x, np.ndarray cost_func, float eta, float U, float L, float D, float tau, np.ndarray previous, float accepted, np.ndarray w, np.ndarray phi, float scale, np.ndarray c):
#     cdef float hit_cost, pseudo_cost_a, pseudo_cost_b, next_accepted
#     hit_cost = (cost_func @ x)
#     next_accepted = (accepted + (c.T @ x))
#     pseudo_cost_a = thresholdAntiDeriv(accepted, U,L,D,tau,eta)
#     pseudo_cost_b = thresholdAntiDeriv(next_accepted, U,L,D,tau,eta)
#     #pseudo_cost = integrate.quad(thresholdFunc, accepted, (accepted + (c.T @ x)), args=(U,L,D,tau,eta))[0]
#     return hit_cost + (weighted_l1_norm(x, previous, phi, w, cvxpy=False) * scale) - (pseudo_cost_b - pseudo_cost_a)#pseudo_cost
#     #return hit_cost + (weighted_l1_norm(x, previous, phi, w, cvxpy=False) * scale) - pseudo_cost

def thresholdAntiDeriv(w, U, L, D, tau, eta):
    return U*w - tau*w + (tau * eta - U * eta + D*eta + U) * cp.exp( w / eta )

def pcmMinimization(x, cost_func, eta, U, L, D, tau, previous, accepted, w, phi, scale, c):
    hit_cost = (cost_func @ x)
    next_accepted = (accepted + (c.T @ x))
    pseudo_cost_a = thresholdAntiDeriv(accepted, U,L,D,tau,eta)
    pseudo_cost_b = thresholdAntiDeriv(next_accepted, U,L,D,tau,eta)
    #pseudo_cost = integrate.quad(thresholdFunc, accepted, (accepted + (c.T @ x)), args=(U,L,D,tau,eta))[0]
    return hit_cost + (weighted_l1_norm(x, previous, phi, w, cvxpy=True) * scale) - (pseudo_cost_b - pseudo_cost_a)#pseudo_cost
    #return hit_cost + (weighted_l1_norm(x, previous, phi, w, cvxpy=False) * scale) - pseudo_cost