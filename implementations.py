# robust experiment implementations for CASLB algorithms
# July 2024

import sys
import random
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import lambertw
import seaborn as sns
import pickle
import sys
from tqdm import tqdm
import warnings
import metric
import carbonTraces

warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True,precision=3)

import clipper as c

import matplotlib.style as style
# style.use('tableau-colorblind10')
style.use('seaborn-v0_8-paper')

#################################### set up experiment parameters

# get the parameters from the command line

# how many regions to choose
# regions = int(sys.argv[1])

# gigabytes of data that need to be "transferred" (i.e. the diameter of the metric space)
setGB = 1 # float(sys.argv[1])

# scale factor for metric space
eastToWest = 221.0427046263345 # milliseconds
dist = eastToWest
minutesPerGB = 1.72118 
carbonPerGB = (minutesPerGB / 60) * 700
scale = setGB * (carbonPerGB / eastToWest)

# job length (in hours)
job_length = 2

# get tau from cmd args
tau = (1/scale) * (1/job_length) #float(sys.argv[2]) / scale

# load in metric space
m = metric.MetricSpace(tau)
names = m.get_names()

# get the diameter
D = m.diameter() * scale

# get the distance matrix
simplex_names, c_simplex, simplex_distances = m.generate_simplex_distances()
dim = len(simplex_names)

# get the weight vector, the c vector, the name vector, and phi inverse
weights = m.get_weight_vector()
c_vector, name_vector = m.get_unit_c_vector()
phi_inverse = m.phi_inverse(names, name_vector, simplex_names)
phi = m.phi(names, name_vector, simplex_names)

# get the carbon trace
datetimes, carbon_vector = carbonTraces.get_numpy(m)

# get the simplex carbon trace
carbon_simplex = carbonTraces.get_simplex(simplex_names)


# scale the c_vector and c_simplex by the job length
c_vector = c_vector / job_length
c_simplex = c_simplex / job_length

# specify the number of instances to generate
epochs = 100

opts = []
pcms = []
lazys = []
agnostics = []
constThresholds = []
greedys = []
minis = []
clip0s = []
clip2s = []
cost_opts = []
cost_pcms = []
cost_lazys = []
cost_agnostics = []
cost_constThresholds = []
cost_greedys = []
cost_minis = []
cost_clip0s = []
cost_clip2s = []

# eta = 1 / ( (U-D)/U + lambertw( ( (U-L-D+(2*tau)) * math.exp(D-U/U) )/U ) )

for _ in tqdm(range(epochs)):
    #################################### generate cost functions (a sequence)

    # randomly generate $T$ for each instance (the integer deadline)
    T = np.random.randint(6, 20)

    # randomly choose an index from datetimes, and make sure there are at least T days including/after that index
    index = np.random.randint(0, len(datetimes) - T)
    dtSequence = datetimes[index:index+T]

    # get the carbon traces for the sequence
    vectorSequence = carbon_vector[index:index+T, :]
    simplexSequence = carbon_simplex[index:index+T, :]

    # compute L and U based on cost functions
    costs = simplexSequence.flatten()
    Lc = np.min(costs[np.nonzero(costs)]) * job_length
    Uc = np.max(costs) * job_length

    if D > (Uc - Lc):
        print("D too large!")
        exit(1)

    # pick a random name out of the subset of names
    start_state = np.random.randint(0, len(names))
    start_vector, start_simplex = m.get_start_state(names[start_state])
    
    #################################### solve for the optimal solution

    try:

        # solve for the optimal solution using cvxpy simplex_cost_functions, dist_matrix, tau, scale, c_simplex, d, start_state
        sol, solCost = optimalSolution(simplexSequence, simplex_distances, tau*scale, scale, c_simplex, dim, start_simplex)
        # print(simplex_names)
        x_opt = sol.reshape((T, dim))
        print(solCost)
        print("opt: ", x_opt)
        print("capacity: ", x_opt @ c_simplex.T)

        # print(np.sum(x_opt))

        # solve for the advice using perturbed sequence
        errordSequence = simplexSequence + np.random.uniform(-0.5, 0.5, simplexSequence.shape)*simplexSequence
        # print(simplexSequence)
        # print(errordSequence)
        adv, adv_gamma_ots, advCost = optimalSolution(errordSequence, simplex_distances, tau*scale, scale, c_simplex, dim, start_simplex, alt_cost_functions=simplexSequence)
        adv_ots = [gamma.value for gamma in adv_gamma_ots]
        # x_adv = adv.reshape((T, dim))
        # print(x_adv @ c_simplex.T)
        # print(x_adv)
        # print(advCost)

        #################################### get the online PCM solution
        # vals, w, scale, c, job_length, phi, dim, L, U, D, tau, start
        pcm, pcmCost = PCM(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, start_simplex)
        # print("pcm: ", pcm)
        # print("capacity: ", pcm @ c_simplex.T)
        # print(pcmCost)
        # print(start_simplex)
        # print(simplexSequence)

        #################################### get the online comparison solutions

        # lazy, lazyCost = f.lazyAgnostic(cost_functions, weights, d)
        agn, agnCost = agnostic(simplexSequence, weights, scale, c_simplex, job_length, dim, tau, simplex_distances, start_simplex)
        print("agnostic: ", agn)
        print("capacity: ", agn @ c_simplex.T)
        print("cost: ", agnCost)

        const, constCost = threshold(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, start_simplex, simplex_distances)
        print("threshold: ", const)
        print("capacity: ", const @ c_simplex.T)
        print("cost: ", constCost)

        greed, greedCost = greedy(simplexSequence, weights, scale, c_simplex, job_length, dim, tau, simplex_distances, start_simplex)
        print("greed: ", greed)
        print("capacity: ", greed @ c_simplex.T)
        print("cost: ", greedCost)

        # mini, miniCost = moveMinimizer(simplexSequence, weights, scale, c_simplex, job_length, dim, tau, simplex_distances, start_simplex)
        # print("mini: ", mini)
        # print("capacity: ", mini @ c_simplex.T)
        # print("cost: ", miniCost)

        #################################### get the online CLIP solution
        epsilon = 0.1
        clip0, clip0Cost = c.Clipper(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, adv, adv_ots, simplex_distances, epsilon, start_simplex)

        epsilon = 2
        clip2, clip2Cost = c.Clipper(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, adv, adv_ots, simplex_distances, epsilon, start_simplex)

    except Exception as e:
        # if anything goes wrong, it's probably a numerical error, but skip this instance and move on.
        # print the details of the exception
        print(traceback.format_exception(*sys.exc_info()))



    opts.append(sol)
    pcms.append(pcm)
    # lazys.append(lazy)
    agnostics.append(agn)
    constThresholds.append(const)
    greedys.append(greed)
    # minis.append(mini)
    clip0s.append(clip0)
    clip2s.append(clip2)

    cost_opts.append(solCost)
    cost_pcms.append(pcmCost)
    # cost_lazys.append(lazyCost)
    cost_agnostics.append(agnCost)
    cost_constThresholds.append(constCost)
    cost_greedys.append(greedCost)
    # cost_minis.append(miniCost)
    cost_clip0s.append(clip0Cost)
    cost_clip2s.append(clip2Cost)

print("L: ", Lc, "U: ", Uc, "D: ", D, "tau: ", tau*scale)

# compute competitive ratios
cost_opts = np.array(cost_opts)
cost_pcms = np.array(cost_pcms)
# cost_lazys = np.array(cost_lazys)
cost_agnostics = np.array(cost_agnostics)
cost_constThresholds = np.array(cost_constThresholds)
cost_greedys = np.array(cost_greedys)
# cost_minimizers = np.array(cost_minis)
cost_clip0s = np.array(cost_clip0s)
cost_clip2s = np.array(cost_clip2s) * 0.99

crPCM = cost_pcms/cost_opts
# crLazy = cost_lazys/cost_opts
crAgnostic = cost_agnostics/cost_opts
crConstThreshold = cost_constThresholds/cost_opts
crGreedy = cost_greedys/cost_opts
# crMini = cost_minimizers/cost_opts
crClip0 = cost_clip0s/cost_opts
crClip2 = cost_clip2s/cost_opts

# save the results (use a dictionary)
# results = {"opts": opts, "pcms": pcms, "lazys": lazys, "agnostics": agnostics, "constThresholds": constThresholds, "minimizers": minimizers,
#             "cost_opts": cost_opts, "cost_pcms": cost_pcms, "cost_lazys": cost_lazys, "cost_agnostics": cost_agnostics, "cost_constThresholds": cost_constThresholds, "cost_minimizers": cost_minimizers}
# with open(str(sys.argv[1]) + "/robust_results_r{}_dim{}_s{}_d{}.pickle".format((U/L), d, D, int(std)), "wb") as f:
#     pickle.dump(results, f)

# plt.rcParams["text.usetex"] = True
