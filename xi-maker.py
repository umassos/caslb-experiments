# experiment implementations for CarbonClipper
# July 2024

import sys
import random
import math
import itertools
import numpy as np
import pandas as pd
import traceback
import ot
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

def experiment(xi):
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
    job_length = 1

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
    epochs = 1500

    opts = []
    advs = []
    pcms = []
    clip0s = []
    clip2s = []
    clip5s = []
    clip10s = []
    agnostics = []
    constThresholds = []
    greedys = []


    cost_opts = []
    cost_advs = []
    cost_pcms = []
    cost_clip0s = []
    cost_clip2s = []
    cost_clip5s = []
    cost_clip10s = []
    cost_agnostics = []
    cost_constThresholds = []
    cost_greedys = []

    # eta = 1 / ( (U-D)/U + lambertw( ( (U-L-D+(2*tau)) * math.exp(D-U/U) )/U ) )

    for _ in range(epochs):
        #################################### generate cost functions (a sequence)

        # randomly generate $T$ for the instance (the integer deadline)
        T = 10

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
        
        try:
            #################################### solve for the optimal solution
            
            # solve for the optimal solution using cvxpy
            sol, solCost = f.optimalSolution(simplexSequence, simplex_distances, tau*scale, scale, c_simplex, dim, start_simplex)
            x_opt = sol.reshape((T, dim))
            # print("sol decisions: {}".format(np.array(sol)))
            # print(( sol @ c_simplex.T) )
            # print("solCost: {}".format(solCost))
            
            # print(x_opt)
            # print(solCost)
            # print(np.sum(x_opt))

            # #################################### get the "bad" solution
            # solve for the adversarial advice
            bad, badCost = f.adversarialSolution(simplexSequence, simplex_distances, tau*scale, scale, c_simplex, dim, start_simplex)
            adv = ((xi)*np.array(bad)).reshape((T, dim)) + (1-xi)*np.array(sol)
            adv_ots = []
            for i in range(T):
                if i == 0:
                    gamma = ot.emd(start_simplex, np.array(adv[i]), simplex_distances)
                    adv_ots.append(gamma)
                gamma = ot.emd(np.array(adv[i-1]), np.array(adv[i]), simplex_distances)
                adv_ots.append(gamma)

            #################################### get the online PCM solution

            pcm, pcmCost = f.PCM(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, start_simplex)
            # print(pcm @ c_simplex.T)
            # print(pcm)
            # print(pcmCost)
            # print(start_simplex)
            # print(simplexSequence)

            #################################### get the online comparison solutions

            # lazy, lazyCost = f.lazyAgnostic(cost_functions, weights, d)
            agn, agnCost = f.agnostic(simplexSequence, weights, scale, c_simplex, job_length, dim, tau, simplex_distances, start_simplex)

            const, constCost = f.threshold(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, start_simplex, simplex_distances)

            greed, greedCost = f.greedy(simplexSequence, weights, scale, c_simplex, job_length, dim, tau, simplex_distances, start_simplex)

            #################################### get the online CLIP solution
            epsilon = 0.1
            clip0, clip0Cost = c.Clipper(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, adv, adv_ots, simplex_distances, epsilon, start_simplex)

            epsilon = 2
            clip2, clip2Cost = c.Clipper(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, adv, adv_ots, simplex_distances, epsilon, start_simplex)

            # if xi = 0, check that the clip2 cost is no more than 3* the optimal cost
            if xi == 0:
                assert clip2Cost <= 3*solCost

            epsilon = 5
            clip5, clip5Cost = c.Clipper(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, adv, adv_ots, simplex_distances, epsilon, start_simplex)

            epsilon = 10
            clip10, clip10Cost = c.Clipper(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, adv, adv_ots, simplex_distances, epsilon, start_simplex)

        except Exception as e:
            # if anything goes wrong, it's probably a numerical error, but skip this instance and move on.
            # print the details of the exception
            print(traceback.format_exception(*sys.exc_info()))
            # solHitSoFar = 0.0
            # clipHitSoFar = 0.0
            # solAccepted = 0.0
            # solSwitchSoFar = 0.0
            # clipSwitchSoFar = 0.0
            # for (i, cost_func) in enumerate(simplexSequence):
            #     print("i: {}".format(i))
            #     solHitSoFar += cost_func @ sol[i]
            #     solAccepted += c_simplex.T @ sol[i]
            #     clipHitSoFar += cost_func @ clip2[i]
            #     if i == 0:
            #         start_emd = np.array(start_simplex) / np.sum(start_simplex)
            #         soli_emd = np.array(sol[i]) / np.sum(sol[i])
            #         clipi_emd = np.array(clip2[i]) / np.sum(clip2[i])
            #         solSwitchSoFar += ot.emd2(start_emd, soli_emd, simplex_distances) * scale
            #         clipSwitchSoFar += ot.emd2(start_emd, clipi_emd, simplex_distances) * scale
            #     else:
            #         soli_emd = np.array(sol[i]) / np.sum(sol[i])
            #         soli1_emd = np.array(sol[i-1]) / np.sum(sol[i-1])
            #         clipi_emd = np.array(clip2[i]) / np.sum(clip2[i])
            #         clipi1_emd = np.array(clip2[i-1]) / np.sum(clip2[i-1])
            #         solSwitchSoFar += ot.emd2(soli1_emd, soli_emd, simplex_distances) * scale
            #         clipSwitchSoFar += ot.emd2(clipi1_emd, clipi_emd, simplex_distances) * scale
            #     if i == len(simplexSequence) - 1:
            #         print("last time step!")
            #         solSwitchSoFar += (c_simplex.T @ sol[i]) * tau*scale
            #         clipSwitchSoFar += (c_simplex.T @ clip2[i]) * tau*scale
            #     hypotheticalSol = (solHitSoFar+solSwitchSoFar) + (1 - solAccepted) * Lc
            #     print("solSoFar: {}".format((solHitSoFar+solSwitchSoFar)))
            #     print("hypotheticalSol: {}".format(hypotheticalSol))
            #     print("clipSoFar: {}".format(clipHitSoFar+clipSwitchSoFar))
            #     if solHitSoFar + solSwitchSoFar > 0:
            #         print("clip/sol so far: {}".format((clipHitSoFar+clipSwitchSoFar)/(solHitSoFar + solSwitchSoFar)))
            #     print("clip/hypotheticalSol: {}".format((clipHitSoFar+clipSwitchSoFar)/hypotheticalSol))
            #     print("")
            # print("length: {}".format(len(simplexSequence)))
            # solHit, solSwitch = f.objectiveSimplexNoOpt(sol, simplexSequence, simplex_distances, scale, dim, c_simplex, tau*scale, start_simplex, cpy=False, debug=True)
            # clipHit, clipSwitch = f.objectiveSimplexNoOpt(clip2, simplexSequence, simplex_distances, scale, dim, c_simplex, tau*scale, start_simplex, cpy=False, debug=True)
            # print("final Sol hit: {}".format(solHit))
            # print("running Sol hit: {}".format(solHitSoFar))
            # print("final Sol switch: {}".format(solSwitch))
            # print("running Sol switch: {}".format(solSwitchSoFar))
            # print("final Clip hit: {}".format(clipHit))
            # print("running Clip hit: {}".format(clipHitSoFar))
            # print("final Clip switch: {}".format(clipSwitch))
            # print("running Clip switch: {}".format(clipSwitchSoFar))
            continue

        opts.append(sol)
        advs.append(adv)
        pcms.append(pcm)
        agnostics.append(agn)
        constThresholds.append(const)
        greedys.append(greed)
        clip0s.append(clip0)
        clip2s.append(clip2)
        clip5s.append(clip5)
        clip10s.append(clip10)

        cost_opts.append(solCost)
        cost_advs.append(badCost)
        cost_pcms.append(pcmCost)
        cost_agnostics.append(agnCost)
        cost_constThresholds.append(constCost)
        cost_greedys.append(greedCost)
        cost_clip0s.append(clip0Cost)
        cost_clip2s.append(clip2Cost)
        cost_clip5s.append(clip5Cost)
        cost_clip10s.append(clip10Cost)


    # compute competitive ratios
    cost_opts = np.array(cost_opts)
    cost_advs = np.array(cost_advs)
    cost_pcms = np.array(cost_pcms)
    cost_agnostics = np.array(cost_agnostics)
    cost_constThresholds = np.array(cost_constThresholds)
    cost_greedys = np.array(cost_greedys)
    cost_clip0s = np.array(cost_clip0s)
    cost_clip2s = np.array(cost_clip2s)
    cost_clip5s = np.array(cost_clip5s)
    cost_clip10s = np.array(cost_clip10s)
    # cost_baseline2s = np.array(cost_baseline2s)

    crPCM = cost_pcms/cost_opts
    crAgnostic = cost_agnostics/cost_opts
    crConstThreshold = cost_constThresholds/cost_opts
    crGreedy = cost_greedys/cost_opts
    crClip0 = cost_clip0s/cost_opts
    crClip2 = cost_clip2s/cost_opts
    crClip5 = cost_clip5s/cost_opts
    crClip10 = cost_clip10s/cost_opts

    # save the results (use a dictionary)
    results = {"opts": opts, "pcms": pcms, "agnostics": agnostics, "constThresholds": constThresholds, "greedys": greedys, "clip0s": clip0s, "clip2s": clip2s, "clip5s": clip5s, "clip10s": clip10s, "advs": advs,
               "cost_opts": cost_opts, "cost_pcms": cost_pcms, "cost_agnostics": cost_agnostics, "cost_constThresholds": cost_constThresholds, "cost_greedys": cost_greedys, "cost_clip0s": cost_clip0s, "cost_clip2s": cost_clip2s, "cost_clip5s": cost_clip5s, "cost_clip10s": cost_clip10s, "cost_advs": cost_advs}
    # results = {"opts": opts, "pcms": pcms, "lazys": lazys, "agnostics": agnostics, "constThresholds": constThresholds, "minimizers": minimizers, "clip2s": clip2s, "baseline2s": baseline2s,
    #             "cost_opts": cost_opts, "cost_pcms": cost_pcms, "cost_lazys": cost_lazys, "cost_agnostics": cost_agnostics, "cost_constThresholds": cost_constThresholds, "cost_minimizers": cost_minimizers, "cost_clip2s": cost_clip2s, "cost_baseline2s": cost_baseline2s}
    with open("xi/xi_results{}.pickle".format(xi*100), "wb") as f:
        pickle.dump(results, f)


    # print mean and 95th percentile of each competitive ratio
    print("Diameter: {}".format(D))
    print("Xi Value: {}".format(xi))
    print("PCM: ", np.mean(crPCM), np.percentile(crPCM, 95))
    print("agnostic: ", np.mean(crAgnostic), np.percentile(crAgnostic, 95))
    print("simple threshold: ", np.mean(crConstThreshold), np.percentile(crConstThreshold, 95))
    print("greedy: ", np.mean(crGreedy), np.percentile(crGreedy, 95))
    print("clip0: ", np.mean(crClip0), np.percentile(crClip0, 95))
    print("clip2: ", np.mean(crClip2), np.percentile(crClip2, 95))
    print("clip5: ", np.mean(crClip5), np.percentile(crClip5, 95))
    print("clip10: ", np.mean(crClip10), np.percentile(crClip10, 95))
    # print("baseline2: ", np.mean(crBaseline2), np.percentile(crBaseline2, 95))
    # print("alpha bound: ", alpha)



# use multiprocessing here
if __name__ == "__main__":
    xis = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
    with Pool(10) as p:
        p.map(experiment, xis)

# if __name__ == "__main__":
#     gbs = [1, 3, 5, 7, 9]
#     for gb in tqdm(gbs):
#         experiment(gb)
