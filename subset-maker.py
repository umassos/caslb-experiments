# experiment implementations for CarbonClipper
# July 2024

import sys
import random
import math
import itertools
import traceback
import numpy as np
import pandas as pd
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

def experiment(subset):
    import implementations as f
    import clipper as c
    #################################### set up experiment parameters

    # get the parameters from the command line
    subset_names, subset_header = subset

    # gigabytes of data that need to be "transferred" (i.e. the diameter of the metric space)
    setGB = 4 # float(sys.argv[1])

    # scale factor for metric space
    eastToWest = 221.0427046263345 # milliseconds
    dist = eastToWest
    minutesPerGB = 1.72118 
    carbonPerGB = (minutesPerGB / 60) * 365
    scale = setGB * (carbonPerGB / eastToWest)

    # job length (in hours)
    job_length = 1

    # get tau from cmd args
    tau = (1/scale) * (1/job_length) #float(sys.argv[2]) / scale

    # load in metric space
    m = metric.MetricSpace(tau, names=subset_names)
    names = subset_names

    # get the diameter
    D = m.diameter(subset_names) * scale

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
    pcms = []
    clip0s = []
    clip2s = []
    agnostics = []
    constThresholds = []
    greedys = []
    delayGreedys = []

    cost_opts = []
    cost_pcms = []
    cost_clip0s = []
    cost_clip2s = []
    cost_agnostics = []
    cost_constThresholds = []
    cost_greedys = []
    cost_delayGreedys = []


    # eta = 1 / ( (U-D)/U + lambertw( ( (U-L-D+(2*tau)) * math.exp(D-U/U) )/U ) )

    for _ in range(epochs):
        #### get a random job length from the cloud traces
        job_length = loadTraces.randomJobLength(1, 10)

        # get tau from cmd args
        tau = (1/scale) * (1/job_length) #float(sys.argv[2]) / scale

        # get the distance matrix
        simplex_names, c_simplex, simplex_distances = m.generate_simplex_distances()

        # get the weight vector, the c vector, the name vector, and phi inverse
        c_vector, name_vector = m.get_unit_c_vector()

        # scale the c_vector and c_simplex by the job length
        c_vector = c_vector / job_length
        c_simplex = c_simplex / job_length
        
        #################################### generate cost functions (a sequence)

        # randomly generate $T$ for the instance (the integer deadline)
        T = np.random.randint(12, 48)

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
            break
        
        # pick a random name out of the subset of names
        start_state = np.random.randint(0, len(names))
        start_vector, start_simplex = m.get_start_state(names[start_state])
        
        try:
            #################################### solve for the optimal solution
            
            # solve for the optimal solution using cvxpy
            sol, solCost = f.optimalSolution(simplexSequence, simplex_distances, tau*scale, scale, c_simplex, dim, start_simplex)
            x_opt = sol.reshape((T, dim))
            # print(sol)
            # print(x_opt)
            # print(solCost)
            # print(np.sum(x_opt))

            # #################################### get the "bad" solution
            # solve for the advice using perturbed sequence
            randomNoise = np.random.uniform(Lc/job_length, Uc/job_length, simplexSequence.shape)
            # wherever simplexSequence is 0, set randomNoise to 0 (OFF states)
            randomNoise[simplexSequence == 0] = 0
            errordSequence = (0.6)*simplexSequence + (0.4)*randomNoise
            # print(simplexSequence)
            # print(errordSequence) (simplex_cost_functions, dist_matrix, tau, scale, c_simplex, d, start_state):
            adv, adv_gamma_ots, advCost = f.optimalSolution(errordSequence, simplex_distances, tau*scale, scale, c_simplex, dim, start_simplex, alt_cost_functions=simplexSequence)
            adv_ots = [gamma.value for gamma in adv_gamma_ots]
            x_adv = adv.reshape((T, dim))

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

            delayGreed, delayGreedCost = f.delayedGreedy(simplexSequence, errordSequence, weights, scale, c_simplex, job_length, dim, tau, simplex_distances, start_simplex)

            #################################### get the online CLIP solution
            epsilon = 0.1
            clip0, clip0Cost = c.Clipper(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, adv, adv_ots, simplex_distances, epsilon, start_simplex)

            epsilon = 2
            clip2, clip2Cost = c.Clipper(simplexSequence, weights, scale, c_simplex, job_length, phi, dim, Lc, Uc, D, tau*scale, adv, adv_ots, simplex_distances, epsilon, start_simplex)

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
        pcms.append(pcm)
        agnostics.append(agn)
        constThresholds.append(const)
        greedys.append(greed)
        delayGreedys.append(delayGreed)
        clip0s.append(clip0)
        clip2s.append(clip2)

        cost_opts.append(solCost)
        cost_pcms.append(pcmCost)
        cost_agnostics.append(agnCost)
        cost_constThresholds.append(constCost)
        cost_greedys.append(greedCost)
        cost_delayGreedys.append(delayGreedCost)
        cost_clip0s.append(clip0Cost)
        cost_clip2s.append(clip2Cost)

    if len(opts) == 0:
        print("D too large. Skipping.")
        return

    # compute competitive ratios
    cost_opts = np.array(cost_opts)
    cost_pcms = np.array(cost_pcms)
    cost_agnostics = np.array(cost_agnostics)
    cost_constThresholds = np.array(cost_constThresholds)
    cost_greedys = np.array(cost_greedys)
    cost_delayGreedys = np.array(cost_delayGreedys)
    cost_clip0s = np.array(cost_clip0s)
    cost_clip2s = np.array(cost_clip2s)
    # cost_baseline2s = np.array(cost_baseline2s)

    crPCM = cost_pcms/cost_opts
    crAgnostic = cost_agnostics/cost_opts
    crConstThreshold = cost_constThresholds/cost_opts
    crGreedy = cost_greedys/cost_opts
    crDelayGreedy = cost_delayGreedys/cost_opts
    crClip0 = cost_clip0s/cost_opts
    crClip2 = cost_clip2s/cost_opts
    # crBaseline2 = cost_baseline2s/cost_opts

    # save the results (use a dictionary)
    results = {"opts": opts, "pcms": pcms, "agnostics": agnostics, "constThresholds": constThresholds, "greedys": greedys, "delayGreedys": delayGreedys, "clip0s": clip0s, "clip2s": clip2s,
               "cost_opts": cost_opts, "cost_pcms": cost_pcms, "cost_agnostics": cost_agnostics, "cost_constThresholds": cost_constThresholds, "cost_greedys": cost_greedys, "cost_delayGreedys": cost_delayGreedys, "cost_clip0s": cost_clip0s, "cost_clip2s": cost_clip2s}
    # results = {"opts": opts, "pcms": pcms, "lazys": lazys, "agnostics": agnostics, "constThresholds": constThresholds, "minimizers": minimizers, "clip2s": clip2s, "baseline2s": baseline2s,
    #             "cost_opts": cost_opts, "cost_pcms": cost_pcms, "cost_lazys": cost_lazys, "cost_agnostics": cost_agnostics, "cost_constThresholds": cost_constThresholds, "cost_minimizers": cost_minimizers, "cost_clip2s": cost_clip2s, "cost_baseline2s": cost_baseline2s}
    with open("subset/subset_{}.pickle".format(subset_header), "wb") as f:
        pickle.dump(results, f)


    # print mean and 95th percentile of each competitive ratio
    print("Diameter: {}".format(D))
    print("Simulated Subset: {}".format(subset_names))
    # if np.mean(crPCM) < np.mean(crGreedy):
    #     #if np.percentile(crPCM, 95) < np.percentile(crGreedy, 95):
    print("PCM: ", np.mean(crPCM), np.percentile(crPCM, 95))
    print("agnostic: ", np.mean(crAgnostic), np.percentile(crAgnostic, 95))
    print("simple threshold: ", np.mean(crConstThreshold), np.percentile(crConstThreshold, 95))
    print("greedy: ", np.mean(crGreedy), np.percentile(crGreedy, 95))
    print("clip0: ", np.mean(crClip0), np.percentile(crClip0, 95))
    print("clip2: ", np.mean(crClip2), np.percentile(crClip2, 95))
    # print("baseline2: ", np.mean(crBaseline2), np.percentile(crBaseline2, 95))
    # print("alpha bound: ", alpha)



# use multiprocessing here
if __name__ == "__main__":
    originalNames = [
        "us-east-1",      # US East (N. Virginia)
        "us-west-1",      # US West (N. California)
        "us-west-2",      # US West (Oregon)
        "af-south-1",     # Africa (Cape Town)
        "ap-south-2",     # Asia Pacific (Hyderabad)
        "ap-northeast-2", # Asia Pacific (Seoul)
        "ap-southeast-2", # Asia Pacific (Sydney)
        "ca-central-1",   # Canada (Central)
        "eu-central-1",   # Europe (Frankfurt)
        "eu-west-2",      # Europe (London)
        "eu-west-3",      # Europe (Paris)
        "eu-north-1",     # Europe (Stockholm)
        "sa-east-1",       # South America (São Paulo)
        "il-central-1"    # Israel (Tel Aviv)
    ]
    GDPRsubset = [ "eu-central-1", "eu-west-2", "eu-west-3",  "eu-north-1" ]
    NAsubset = ['us-east-1', 'us-west-1', 'us-west-2', 'ca-central-1']
    # crossingsSubset = ["us-east-1", "us-west-2",  "af-south-1",  "ap-south-2",  "ap-northeast-2", "ap-southeast-2", "eu-central-1", "eu-west-2", "il-central-1" ]
    # crossings2Subset = [ "us-east-1", "us-west-1", "us-west-2", "af-south-1", "ap-south-2", "ap-northeast-2", "ap-southeast-2", "eu-central-1", "eu-west-2", "il-central-1"]
    noHydroSubset = ["us-east-1", "us-west-1", "us-west-2",  "af-south-1", "ap-south-2",  "ap-northeast-2", "ap-southeast-2", "eu-central-1", "eu-west-2", "eu-west-3", "sa-east-1", "il-central-1" ]

    candidateSubset = ['af-south-1', 'us-east-1', 'us-west-2', 'us-west-1', 'ap-northeast-2', 'eu-west-3'] # twice!
    candidate2Subset = ['sa-east-1', 'eu-west-3', 'us-west-2', 'ap-south-2', 'ap-southeast-2'] # three times good margin
    candidate3Subset = ['us-west-1', 'ap-northeast-2', 'eu-central-1', 'ap-south-2', 'il-central-1', 'eu-north-1', 'af-south-1'] # twice!
    candidate4Subset = ['ca-central-1', 'ap-southeast-2', 'af-south-1', 'il-central-1', 'ap-south-2'] # twice!
    candidate5Subset = ['ap-northeast-2', 'il-central-1', 'af-south-1', 'eu-west-3', 'us-west-2'] # twice, bad margin
    candidate6Subset = ['us-east-1', 'ap-south-2', 'eu-north-1', 'us-west-2', 'us-west-1', 'af-south-1']
    candidate7Subset = ['us-west-1', 'eu-central-1', 'ap-south-2', 'eu-west-3', 'eu-north-1', 'ap-southeast-2'] # twice!
    candidate8Subset = ['ap-southeast-2', 'us-west-2', 'eu-central-1', 'ca-central-1', 'il-central-1', 'ap-northeast-2'] # twice!
    candidate9Subset = ['eu-central-1', 'ap-northeast-2', 'eu-west-2', 'us-west-1', 'ca-central-1', 'ap-southeast-2', 'ap-south-2'] # three ok margin
    candidate10Subset = ['us-west-1', 'eu-west-2', 'ap-south-2', 'ap-northeast-2', 'us-east-1', 'eu-north-1'] # twice good margin
    candidate11Subset = ['ap-northeast-2', 'us-east-1', 'ap-southeast-2', 'ca-central-1', 'eu-west-3'] # three times good margin
    candidate12Subset = ['il-central-1', 'us-west-1', 'sa-east-1', 'af-south-1', 'eu-west-2', 'eu-west-3', 'us-east-1'] # twice ok margin
    candidate13Subset = ['eu-west-3', 'ca-central-1', 'us-east-1', 'il-central-1', 'ap-northeast-2', 'sa-east-1', 'ap-southeast-2']
    candidate14Subset = ['eu-west-3', 'af-south-1', 'eu-central-1', 'il-central-1', 'ap-northeast-2', 'us-west-1'] # twice bad margin

    candidates = [candidateSubset, candidate2Subset, candidate3Subset, candidate4Subset, candidate5Subset, candidate6Subset, candidate7Subset, candidate8Subset, candidate9Subset, candidate10Subset, candidate11Subset, candidate12Subset, candidate13Subset, candidate14Subset]
    candidate_names = ["candidate1", "candidate2", "candidate3", "candidate4", "candidate5", "candidate6", "candidate7", "candidate8", "candidate9", "candidate10", "candidate11", "candidate12", "candidate13", "candidate14"]
    
    
    subsets = [(GDPRsubset, "GDPR"), (NAsubset, "NA"), (noHydroSubset, "noHydro")]
    # extend subsets with the candidate subsets
    for i in range(len(candidates)):
        subsets.append((candidates[i], candidate_names[i]))
    # for i in range(5, 11):
    #     # choose a number of regions (random number between 5 and 14)
    #     numregions = random.randint(6, 10)
    #     subsets.append((random.sample(originalNames, numregions), "random{}".format(i)))

    with Pool(10) as p:
        p.map(experiment, subsets)
