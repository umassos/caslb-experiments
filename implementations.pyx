import random
import math
from scipy.optimize import linprog
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import scipy.integrate as integrate
from scipy.special import lambertw
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
def weighted_l1_norm(vector1, vector2, weights, cvxpy=False):
    if cvxpy:
        weighted_diff = cp.multiply(cp.abs(vector1 - vector2), weights)
        weighted_sum = cp.sum(weighted_diff)
    else:
        assert vector1.shape == vector2.shape == weights.shape, "Input arrays must have the same shape."

        weighted_diff = np.abs(vector1 - vector2) * weights
        weighted_sum = np.sum(weighted_diff)

    return weighted_sum

# objectiveFunction computes the minimization objective function for the OCS problem.
# vars is the time series of decision variables (dim d x T)
# vals is the time series of cost functions (dim d x T)
# w is the weight of the switching cost 
# dim is the dimension
def objectiveFunction(vars, vals, w, dim, cpy=False):
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
            cost += weighted_l1_norm(vars[i], np.zeros(dim), w, cvxpy=cpy)
        elif i == n-1:
            cost += weighted_l1_norm(vars[i], vars[i-1], w, cvxpy=cpy)
            cost += weighted_l1_norm(np.zeros(dim), vars[i], w, cvxpy=cpy)
        else:
            cost += weighted_l1_norm(vars[i], vars[i-1], w, cvxpy=cpy)
    return cost

def negativeObjectiveFunction(vars, vals, w, dim):
    return -1 * objectiveFunction(vars, vals, w, dim)


# computing the optimal solution
def optimalSolution(cost_functions, weights, d):
    T = len(cost_functions)
    # declare variables
    x = cp.Variable((T, d))
    constraints = [0 <= x, x <= 1]
    # add deadline constraint
    constraints += [cp.sum(x) == 1]
    prob = cp.Problem(cp.Minimize(objectiveFunction(x, cost_functions, weights, d, cpy=True)), constraints)
    prob.solve()
    # print("status:", prob.status)
    # print("optimal value", prob.value)
    # print("optimal var", x.value)
    if prob.status == 'optimal':
        return x.value, prob.value
    else:
        return x.value, 0.0
    

# cpdef tuple[np.ndarray, float] optimalSolution(list vals, np.ndarray w, int dim):
#     cdef int n
#     cdef list all_bounds, b, row, A
#     cdef np.ndarray x0, xstar

#     n = len(vals)
#     all_bounds = [(0,1) for _ in range(0, n*dim)]

#     # declare inequality constraint matrix (2n + 1) x (2n + 1)
#     # and the inequality constraint vector (2n + 1)
#     A = []
#     b = []
#     # append first row (deadline constraint)
#     row = [0 for i in range(0, n*dim)]
#     for i in range(0, n*dim):
#         row[i] = 1
#     A.append(row)
#     b.append(1)

#     x0 = np.ones(n*dim)
#     try:
#         xstar = minimize(objectiveFunction, x0=x0, args=(vals, w, dim), bounds=all_bounds, constraints=LinearConstraint(A, lb=b, ub=b)).x
#     except:
#         print("something went wrong here")
#         xstar = minimize(objectiveFunction, x0=np.zeros(n*dim), args=(vals, w, dim), bounds=all_bounds, constraints=LinearConstraint(A, lb=b, ub=b)).x
#         return xstar, objectiveFunction(xstar, vals, w, dim)
#     else:
#         return xstar, objectiveFunction(xstar, vals, w, dim)


# computing the adversarial solution
cpdef tuple[np.ndarray, float] adversarialSolution(list vals, np.ndarray w, int dim):
    cdef int n
    cdef list all_bounds, b, row, A
    cdef np.ndarray x0, xstar

    n = len(vals)
    all_bounds = [(0,1) for _ in range(0, n*dim)]

    # declare inequality constraint matrix (2n + 1) x (2n + 1)
    # and the inequality constraint vector (2n + 1)
    A = []
    b = []
    # append first row (deadline constraint)
    row = [0 for i in range(0, n*dim)]
    for i in range(0, n*dim):
        row[i] = 1
    A.append(row)
    b.append(1)

    x0 = np.ones(n*dim)
    try:
        xstar = minimize(negativeObjectiveFunction, x0=x0, args=(vals, w, dim), bounds=all_bounds, constraints=LinearConstraint(A, lb=b, ub=b)).x
    except:
        print("something went wrong here")
        xstar = minimize(negativeObjectiveFunction, x0=np.zeros(n*dim), args=(vals, w, dim), bounds=all_bounds, constraints=LinearConstraint(A, lb=b, ub=b)).x
        return xstar, objectiveFunction(xstar, vals, w, dim)
    else:
        return xstar, objectiveFunction(xstar, vals, w, dim)



cpdef float singleObjective(np.ndarray x, np.ndarray cost_func, np.ndarray previous, np.ndarray w):
    return np.dot(cost_func, x) + weighted_l1_norm(x, previous, w) + weighted_l1_norm(np.zeros_like(x), x, w)

# RORO algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
# L                         -- L
# U                         -- U
cpdef tuple[list, float] RORO(list vals, np.ndarray w, int dim, float L, float U):
    cdef int i
    cdef list sol, all_bounds, b
    cdef float cost, alpha, accepted
    cdef np.ndarray previous, A, x0, x_T, x_t, cost_func

    sol = []
    accepted = 0.0

    # get value for beta
    beta = np.max(w)

    # get value for alpha
    alpha = 1 / (1 - (2*beta/U) + lambertw( ( ( (2*beta/U) + (L/U) - 1 ) * math.exp(2*beta/U) ) / math.e ) )

    #simulate behavior of online algorithm using a for loop
    for (i, cost_func) in enumerate(vals):
        if accepted >= 1:
            sol.append(np.zeros(dim))
            continue
        
        remainder = (1 - accepted)
        
        if i == len(vals) - 1: # must accept last cost function
            # get the best x_T which satisfies c(x_T) = remainder
            all_bounds = [(0,1) for _ in range(0, dim)]

            previous = np.zeros(dim)
            if i != 0:
                previous = sol[-1]
            
            A = np.ones(dim)
            b = [remainder]
            constraint = LinearConstraint(A, lb=b, ub=b)

            x0 = np.zeros(dim)
            x_T = minimize(singleObjective, x0=x0, args=(cost_func, previous, w), bounds=all_bounds, constraints=constraint).x

            sol.append(x_T)
            accepted += np.linalg.norm(x_T, ord=1)
            break

        # solve for pseudo cost-defined solution
        previous = np.zeros(dim)
        if i != 0:
            previous = sol[i-1]
        x_t = roroHelper(cost_func, accepted, alpha, L, U, beta, previous, w, dim)

        # print(np.dot(cost_func,x_t) + weighted_l1_norm(x_t, previous, w))
        # print(integrate.quad(thresholdFunc, 0, (0 + np.linalg.norm(x_t, ord=1)), args=(U,L,beta,alpha))[0])

        accepted += np.linalg.norm(x_t, ord=1)
        sol.append(x_t)

    cost = objectiveFunction(np.array(sol), vals, w, dim)
    return sol, cost

# helper for RORO algorithm
cpdef np.ndarray roroHelper(np.ndarray cost_func, float accepted, float alpha, float L, float U, float beta, np.ndarray previous, np.ndarray w, int dim):
    cdef np.ndarray target, x0, A
    cdef list all_bounds, b
    try:
        x0 = np.zeros(dim)
        all_bounds = [(0,1-accepted) for _ in range(0, dim)]
        A = np.ones(dim)
        b = [1]
        constraint = LinearConstraint(A, ub=b)
        target = minimize(roroMinimization, x0=x0, args=(cost_func, alpha, U, L, beta, previous, accepted, w), bounds=all_bounds, constraints=constraint).x
    except:
        print("something went wrong here w_j={}".format(accepted))
        return np.zeros(dim)
    else:
        return target

cpdef float thresholdFunc(float w, float U, float L, float beta, float alpha):
    return U - beta + (U / alpha - U + 2 * beta) * np.exp( w / alpha )

cpdef float roroMinimization(np.ndarray x, np.ndarray cost_func, float alpha, float U, float L, float beta, np.ndarray previous, float accepted, np.ndarray w):
    cdef float hit_cost, pseudo_cost
    hit_cost = np.dot(cost_func, x)
    pseudo_cost = integrate.quad(thresholdFunc, accepted, (accepted + np.linalg.norm(x, ord=1)), args=(U,L,beta,alpha))[0]
    return hit_cost + weighted_l1_norm(x, previous, w) - pseudo_cost


# "lazy agnostic" algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
cpdef tuple[list, float] lazyAgnostic(list vals, np.ndarray w, int dim):
    cdef list sol
    cdef float cost
    cdef np.ndarray x_t

    # choose a the minimum dimension at time 0
    min_dim = np.argmin(vals[0])

    # construct a solution which ramps up to 1/T on the selected dimension
    sol = []
    for _ in vals:
        x_t = np.zeros(dim)
        x_t[min_dim] = 1.0 / len(vals)
        sol.append(x_t)

    cost = objectiveFunction(np.array(sol), vals, w, dim)
    return sol, cost


# "simple agnostic" algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
cpdef tuple[list, float] agnostic(list vals, np.ndarray w, int dim):
    cdef int i
    cdef list sol
    cdef float cost
    cdef np.ndarray x_t

    # choose a the minimum dimension at time 0
    min_dim = np.argmin(vals[0])

    # construct a solution which ramps up to 1 on the selected dimension at the first time step
    sol = []
    for i, _ in enumerate(vals):
        x_t = np.zeros(dim)
        if i == 0:
            x_t[min_dim] = 1.0
        sol.append(x_t)

    cost = objectiveFunction(np.array(sol), vals, w, dim)
    return sol, cost


# "move to minimizer" algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
cpdef tuple[list, float] moveMinimizer(list vals, np.ndarray w, int dim):
    cdef int i
    cdef list sol
    cdef float cost
    cdef np.ndarray x_t

    # construct a solution which ramps up to 1/T on the minimum dimension at each step
    sol = []
    for val in vals:
        x_t = np.zeros(dim)
        min_dim = np.argmin(val)
        x_t[min_dim] = 1.0 / len(vals)
        sol.append(x_t)

    cost = objectiveFunction(np.array(sol), vals, w, dim)
    return sol, cost



# "simple threshold" algorithm implementation
# list of costs (values)    -- vals
# switching cost weight     -- w
# dimension                 -- dim
# L                         -- L
# U                         -- U
cpdef tuple[list, float] threshold(list vals, np.ndarray w, int dim, float L, float U):
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
        accepted += np.linalg.norm(x_t, ord=1)

    cost = objectiveFunction(np.array(sol), vals, w, dim)
    return sol, cost