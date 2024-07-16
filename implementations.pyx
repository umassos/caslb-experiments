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

cimport numpy as np
np.import_array()


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
    

def negativeObjectiveSimplex(vars, vals, dist_matrix, dim, start_state, cpy=False):
    return -1 * objectiveFunctionSimplex(vars, vals, dist_matrix, dim, start_state, cpy=cpy)



# computing the optimal solution
def optimalSolution(simplex_cost_functions, dist_matrix, scale, c_simplex, d, start_state):
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
    prob.solve()
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", x.value)
    if prob.status == 'optimal':
        return x.value, prob.value
    else:
        return x.value, 0.0


# computing the adversarial solution using cvxpy instead
def adversarialSolution(simplex_cost_functions, dist_matrix, c_simplex, d, start_state):
    T = len(simplex_cost_functions)
    # declare variables
    # d here should be like 2n
    x = cp.Variable((T, d))
    constraints = [0 <= x, x <= 1]
    # add deadline constraint
    c = np.array(c_simplex)
    constraints += [np.sum(x @ c.T) == 1]
    prob = cp.Problem(cp.Minimize(negativeObjectiveSimplex(x, simplex_cost_functions, d, cpy=True)), constraints)
    prob.solve()
    # print("status:", prob.status)
    # print("optimal value", prob.value)
    # print("optimal var", x.value)
    if prob.status == 'optimal':
        return x.value, prob.value
    else:
        return x.value, 0.0

# def singleObjective(x, cost_func, previous, w, scale):
#     return np.dot(cost_func, x) + weighted_l1_norm(x, previous, w) + weighted_l1_norm(np.zeros_like(x), x, w) * scale
def singleObjective(x, cost_func, previous, phi, w, scale, start, cpy=True):
    return (cost_func @ x) + weighted_l1_norm(x, previous, phi, w, cvxpy=cpy) * scale + weighted_l1_norm(start, x, phi, w, cvxpy=cpy) * scale

# PCM algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
# L                         -- L
# U                         -- U
# D                         -- diameter of the metric
cpdef tuple[list, float] PCM(np.ndarray vals, np.ndarray w, float scale, np.ndarray c, np.ndarray phi, int dim, float L, float U, float D, float tau, np.ndarray start):
    cdef int i
    cdef list sol, all_bounds, b
    cdef float cost, eta, accepted
    cdef np.ndarray previous, A, x0, x_T, x_t, cost_func

    sol = []
    accepted = 0.0

    # get value for eta
    eta = 1 / ( (U-D)/U + lambertw( ( (D+L-U+(2*tau*scale)) * math.exp((D-U)/U) )/U ) )

    #simulate behavior of online algorithm using a for loop
    for (i, cost_func) in enumerate(vals):
        if accepted >= 1:
            sol.append(start)
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
        x_t = x_t / np.sum(x_t)

        # print(np.dot(cost_func,x_t) + weighted_l1_norm(x_t, previous, w))
        # print(integrate.quad(thresholdFunc, 0, (0 + np.linalg.norm(x_t, ord=1)), args=(U,L,D,eta))[0])

        accepted += c.T @ x_t
        sol.append(x_t) 

    cost = objectiveFunctionTree(np.array(sol), vals, w, scale, dim, start, phi)
    return sol, cost

# helper for PCM algorithm
cpdef np.ndarray pcmHelper(np.ndarray cost_func, float accepted, float eta, float L, float U, float D, float tau, np.ndarray previous, np.ndarray w, float scale, np.ndarray c, np.ndarray phi, int dim, np.ndarray start):
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
    constraints = [0 <= x, x <= 1, cp.sum(x) == 1]
    prob = cp.Problem(cp.Minimize(pcmMinimization(x, cost_func, eta, U, L, D, tau, previous, accepted, w, phi, scale, c)), constraints)
    prob.solve(solver=cp.CLARABEL)
    target = x.value
    if np.sum(target) > 1.01:
        print("sum of target is greater than 1!  sum: {}".format(sum(target)))
    return target

cpdef float thresholdFunc(float w, float U, float L, float D, float tau, float eta):
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



# "lazy agnostic" algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
cpdef tuple[list, float] lazyAgnostic(list vals, np.ndarray w, np.ndarray c, int dim):
    cdef list sol
    cdef float cost
    cdef np.ndarray x_t

    # choose the minimum dimension at time 0 with nonzero C
    # FIX HERE
    min_dim = np.argmin(vals[0])

    # construct a solution which ramps up to 1/T on the selected dimension
    sol = []
    for _ in vals:
        x_t = np.zeros(dim)
        x_t[min_dim] = 1.0 / len(vals)
        sol.append(x_t)

    cost = objectiveFunctionSimplex(np.array(sol), vals, w, dim)
    return sol, cost


# "simple agnostic" algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
cpdef tuple[list, float] agnostic(list vals, np.ndarray w, np.ndarray c, int dim):
    cdef int i
    cdef list sol
    cdef float cost
    cdef np.ndarray x_t

    # choose a the minimum dimension at time 0 with nonzero C
    # FIX HERE
    min_dim = np.argmin(vals[0])

    # construct a solution which ramps up to 1 on the selected dimension at the first time step
    sol = []
    for i, _ in enumerate(vals):
        x_t = np.zeros(dim)
        if i == 0:
            x_t[min_dim] = 1.0
        sol.append(x_t)

    cost = objectiveFunctionSimplex(np.array(sol), vals, w, dim)
    return sol, cost


# "move to minimizer" algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
cpdef tuple[list, float] moveMinimizer(list vals, np.ndarray w, np.ndarray c, int dim):
    cdef int i
    cdef list sol
    cdef float cost
    cdef np.ndarray x_t

    # construct a solution which ramps up to 1/T on the minimum dimension at each step
    # MIN DIM WITH NONZERO C fix here
    sol = []
    for val in vals:
        x_t = np.zeros(dim)
        min_dim = np.argmin(val)
        x_t[min_dim] = 1.0 / len(vals)
        sol.append(x_t)

    cost = objectiveFunctionSimplex(np.array(sol), vals, w, dim)
    return sol, cost



# "simple threshold" algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
# L                         -- L
# U                         -- U
cpdef tuple[list, float] threshold(list vals, np.ndarray w, np.ndarray c, int dim, float L, float U):
    cdef list sol
    cdef float cost, accepted
    cdef np.ndarray x_t

    accepted = 0.0
    threshold = np.sqrt(L*U)

    sol = []
    for i, cost_func in enumerate(vals):
        x_t = np.zeros(dim)

        if accepted >= 1:
            sol.append(x_t)
            continue

        if i == len(vals) - 1: # must accept last cost function
            min_dim = np.argmin(cost_func)
            x_t[min_dim] = 1.0
            sol.append(x_t)
            break
        
        thresholding = cost_func <= threshold
        # in the first location where thresholding is true, we can set the value to 1
        for j in range(0, dim):
            if thresholding[j]:
                x_t[j] = 1.0
                break
        sol.append(x_t)
        accepted += weighted_l1_capacity(x_t, c)

    cost = objectiveFunctionSimplex(np.array(sol), vals, w, dim)
    return sol, cost