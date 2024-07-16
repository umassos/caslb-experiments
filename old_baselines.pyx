
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