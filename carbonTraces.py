# loading carbon traces for CASLB experiments
# Adam Lechowicz
# Jul 2024

import pandas as pd
import numpy as np

# AWS regions
names = [
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

dfs = {}

# Load the carbon traces into a dict of pandas dataframes
for name in names:
    df = pd.read_csv(f"carbon-data/{name}.csv", parse_dates=["datetime"])
    # keep only the columns we need
    df = df[["datetime", "carbon_intensity_avg"]]
    dfs[name] = df

# given a vector of names from the metric, return a numpy matrix and datetime index for the carbon traces
def get_numpy(metric):
    # get the "column names" of the vectors for the tree embedding in the metric space
    name_vector = metric.name_vector.copy()

    for i, name in enumerate(name_vector):
        if "[" in name: 
            name_vector[i] = "none" 
        if "OFF" in name:
            name_vector[i] = "none"
        if name == 'root':
            name_vector[i] = "none" 

    # create a numpy matrix where each column is the corresponding carbon intensity trace for that region
    X = np.zeros((len(dfs["us-east-1"]), len(name_vector)))

    for i, name in enumerate(name_vector):
        if name == "none":
            continue
        X[:, i] = dfs[name]["carbon_intensity_avg"].values

    # print the first few rows of the matrix
    # print(X[:5, :])

    # save the datetimes to a separate pandas series
    datetimes = dfs["us-east-1"]["datetime"]

    # print(datetimes.head())

    return datetimes, X


