# experiment implementations for CarbonClipper
# July 2024

import sys
import random
import math
import itertools
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
from scipy.special import lambertw
from multiprocessing import Pool
import seaborn as sns
import pickle
import sys
from tqdm import tqdm
import warnings
import metric
import carbonTraces

warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True,precision=3)

import matplotlib.style as style
style.use('tableau-colorblind10')
# style.use('seaborn-v0_8-paper')

def experiment(times):
    import implementations as f
    import clipper as c
    #################################### set up experiment parameters

    # get the parameters from the command line

    # how many regions to choose
    # regions = int(sys.argv[1])

    # gigabytes of data that need to be "transferred" (i.e. the diameter of the metric space)
    setGB = 4 # float(sys.argv[1])

    # scale factor for metric space
    eastToWest = 221.0427046263345 # milliseconds
    dist = eastToWest
    minutesPerGB = 1.72118 
    carbonPerGB = (minutesPerGB / 60) * 700
    scale = setGB * (carbonPerGB / eastToWest)

    # job length (in hours)
    job_length = 4

    # get tau from cmd args
    tau = (1/scale) * (1/job_length) #float(sys.argv[2]) / scale

    # load in metric space
    m = metric.MetricSpace(tau)
    names = m.get_names()

    # get the diameter
    D = m.diameter() * scale

    # get the ground truth simplex distances
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
    epochs = 15

    opts = []
    advs = []
    pcms = []
    clip0s = []
    clip2s = []
    agnostics = []
    constThresholds = []
    greedys = []


    cost_opts = []
    cost_advs = []
    cost_pcms = []
    cost_clip0s = []
    cost_clip2s = []
    cost_agnostics = []
    cost_constThresholds = []
    cost_greedys = []


    # eta = 1 / ( (U-D)/U + lambertw( ( (U-L-D+(2*tau)) * math.exp(D-U/U) )/U ) )

    for _ in range(epochs):
        #################################### generate cost functions (a sequence)

        # randomly generate $T$ for the instance (the integer deadline)
        T = 24

        # randomly choose an index from datetimes, and make sure there are at least T days including/after that index
        index = np.random.randint(0, len(datetimes) - T)
        dtSequence = datetimes[index:index+T]

        # randomly generate distance functions
        # get times - 1 additional random simplex distances
        simplex_distance_rands = [simplex_distances] 
        phi_rands = [phi]
        for _ in range(times-1):
            rand_distances, _, rand_phi = m.shuffled_distances()
            simplex_distance_rands.append(rand_distances)
            phi_rands.append(rand_phi)

        # format these distances into a list of length exactly T
        simplex_distance_list = []
        phi_list = []
        repeat = T // times
        for distances in simplex_distance_rands:
            simplex_distance_list.extend([distances] * repeat)
            phi_list.extend([phi] * repeat)
        # add the remaining distances  
        simplex_distance_list.extend([simplex_distance_list[-1]] * (T - len(simplex_distance_list)))
        phi_list.extend([phi_list[-1]] * (T - len(phi_list)))

        # print simplex_distance_list
        print(simplex_distance_list)

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
        
        try:
            #################################### solve for the optimal solution
            
            # solve for the optimal solution using cvxpy
            sol, solCost = f.optimalSolution(simplexSequence, simplex_distance_list, tau*scale, scale, c_simplex, dim, start_simplex, time_varying=True)
            x_opt = sol.reshape((T, dim))
            # print(sol)
            # print(x_opt)
            # print(solCost)
            # print(np.sum(x_opt))

            # #################################### get the "bad" solution
            # solve for the advice using perturbed sequence
            errordSequence = simplexSequence + np.random.uniform(-0.5, 0.5, simplexSequence.shape)*simplexSequence
            # print(simplexSequence)
            # print(errordSequence) (simplex_cost_functions, dist_matrix, tau, scale, c_simplex, d, start_state):
            adv, adv_gamma_ots, advCost = f.optimalSolution(errordSequence, simplex_distance_list, tau*scale, scale, c_simplex, dim, start_simplex, time_varying=True, alt_cost_functions=simplexSequence)
            adv_ots = [gamma.value for gamma in adv_gamma_ots]
            x_adv = adv.reshape((T, dim))

            #################################### get the online PCM solution

            pcm, pcmCost = f.PCM(simplexSequence, weights, scale, c_simplex, job_length, phi_list, dim, Lc, Uc, D, tau*scale, start_simplex, time_varying=True)
            # print(pcm @ c_simplex.T)
            # print(pcm)
            # print(pcmCost)
            # print(start_simplex)
            # print(simplexSequence)

            #################################### get the online comparison solutions

            # lazy, lazyCost = f.lazyAgnostic(cost_functions, weights, d)
            agn, agnCost = f.agnostic(simplexSequence, weights, scale, c_simplex, job_length, dim, tau, simplex_distance_list, start_simplex, time_varying=True)

            const, constCost = f.threshold(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, start_simplex, simplex_distance_list, time_varying=True)

            greed, greedCost = f.greedy(simplexSequence, weights, scale, c_simplex, job_length, dim, tau, simplex_distance_list, start_simplex, time_varying=True)

            #################################### get the online CLIP solution
            epsilon = 0.1
            clip0, clip0Cost = c.Clipper(simplexSequence, weights, scale, c_simplex, job_length, phi_list, dim, Lc, Uc, D, tau*scale, adv, adv_ots, simplex_distance_list, epsilon, start_simplex, time_varying=True)

            epsilon = 2
            clip2, clip2Cost = c.Clipper(simplexSequence, weights, scale, c_simplex, job_length, phi_list, dim, Lc, Uc, D, tau*scale, adv, adv_ots, simplex_distance_list, epsilon, start_simplex, time_varying=True)

            # epsilon = 5
            # clip5, clip5Cost = c.CLIP(cost_functions, weights, d, Lc, Uc, adv, epsilon)

            # epsilon = 10
            # clip10, clip10Cost = c.CLIP(cost_functions, weights, d, Lc, Uc, adv, epsilon)

        except Exception as e:
            # if anything goes wrong, it's probably a numerical error, but skip this instance and move on.
            # print the details of the exception
            print(traceback.format_exception(*sys.exc_info()))
            continue

        opts.append(sol)
        advs.append(adv)
        pcms.append(pcm)
        agnostics.append(agn)
        constThresholds.append(const)
        greedys.append(greed)
        clip0s.append(clip0)
        clip2s.append(clip2)

        cost_opts.append(solCost)
        cost_advs.append(advCost)
        cost_pcms.append(pcmCost)
        cost_agnostics.append(agnCost)
        cost_constThresholds.append(constCost)
        cost_greedys.append(greedCost)
        cost_clip0s.append(clip0Cost)
        cost_clip2s.append(clip2Cost)


    # compute competitive ratios
    cost_opts = np.array(cost_opts)
    cost_advs = np.array(cost_advs)
    cost_pcms = np.array(cost_pcms)
    cost_agnostics = np.array(cost_agnostics)
    cost_constThresholds = np.array(cost_constThresholds)
    cost_greedys = np.array(cost_greedys)
    cost_clip0s = np.array(cost_clip0s)
    cost_clip2s = np.array(cost_clip2s)
    # cost_baseline2s = np.array(cost_baseline2s)

    crAdv = cost_advs/cost_opts
    crPCM = cost_pcms/cost_opts
    crAgnostic = cost_agnostics/cost_opts
    crConstThreshold = cost_constThresholds/cost_opts
    crGreedy = cost_greedys/cost_opts
    crClip0 = cost_clip0s/cost_opts
    crClip2 = cost_clip2s/cost_opts
    # crBaseline2 = cost_baseline2s/cost_opts

    # save the results (use a dictionary)
    results = {"opts": opts, "pcms": pcms, "agnostics": agnostics, "constThresholds": constThresholds, "greedys": greedys, "clip0s": clip0s, "clip2s": clip2s, 
               "cost_opts": cost_opts, "cost_pcms": cost_pcms, "cost_agnostics": cost_agnostics, "cost_constThresholds": cost_constThresholds, "cost_greedys": cost_greedys, "cost_clip0s": cost_clip0s, "cost_clip2s": cost_clip2s}
    # results = {"opts": opts, "pcms": pcms, "lazys": lazys, "agnostics": agnostics, "constThresholds": constThresholds, "minimizers": minimizers, "clip2s": clip2s, "baseline2s": baseline2s,
    #             "cost_opts": cost_opts, "cost_pcms": cost_pcms, "cost_lazys": cost_lazys, "cost_agnostics": cost_agnostics, "cost_constThresholds": cost_constThresholds, "cost_minimizers": cost_minimizers, "cost_clip2s": cost_clip2s, "cost_baseline2s": cost_baseline2s}
    with open("volatility/volatility_results{}.pickle".format(setGB), "wb") as f:
        pickle.dump(results, f)


    # print mean and 95th percentile of each competitive ratio
    print("Diameter: {}".format(D))
    print("Num Swaps: {}".format(times))
    print("PCM: ", np.mean(crPCM), np.percentile(crPCM, 95))
    print("agnostic: ", np.mean(crAgnostic), np.percentile(crAgnostic, 95))
    print("simple threshold: ", np.mean(crConstThreshold), np.percentile(crConstThreshold, 95))
    print("greedy: ", np.mean(crGreedy), np.percentile(crGreedy, 95))
    print("clip0: ", np.mean(crClip0), np.percentile(crClip0, 95))
    print("clip2: ", np.mean(crClip2), np.percentile(crClip2, 95))
    print("Advice: ", np.mean(crAdv), np.percentile(crAdv, 95))
    # print("baseline2: ", np.mean(crBaseline2), np.percentile(crBaseline2, 95))
    # print("alpha bound: ", alpha)



# use multiprocessing here
if __name__ == "__main__":
    times_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    with Pool(1) as p:
        p.map(experiment, times_list)

# if __name__ == "__main__":
#     gbs = [1, 3, 5, 7, 9]
#     for gb in tqdm(gbs):
#         experiment(gb)
