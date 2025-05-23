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

marginal_names = [
    "us-east-1",      # US East (N. Virginia)
    "us-west-1",      # US West (N. California)
    "us-west-2",      # US West (Oregon)
    "ap-southeast-2", # Asia Pacific (Sydney)
    "ca-central-1",   # Canada (Central)
    "eu-central-1",   # Europe (Frankfurt)
    "eu-west-2",      # Europe (London)
    "eu-west-3",      # Europe (Paris)
    "eu-north-1",     # Europe (Stockholm)
]

avg_dfs = {}
mar_dfs = {}

# Load the carbon traces into a dict of pandas dataframes
for name in names:
    df = pd.read_csv(f"carbon-data/{name}.csv", parse_dates=["datetime"])
    # keep only the columns we need
    df = df[["datetime", "carbon_intensity_avg"]]
    avg_dfs[name] = df

# Load the marginal carbon traces into a dict of pandas dataframes
for name in marginal_names:
    df = pd.read_csv(f"marginal-data/{name}.csv", parse_dates=["datetime"])
    # keep only the columns we need
    df = df[["datetime", "marginal_carbon_avg", "marginal_forecast_avg"]]
    # anything below 48 is a data error, truncate it to 48
    df["marginal_carbon_avg"] = np.maximum(df["marginal_carbon_avg"], 48)
    df["marginal_forecast_avg"] = np.maximum(df["marginal_forecast_avg"], 48)
    mar_dfs[name] = df

# given a vector of names from the metric, return a numpy matrix and datetime index for the carbon traces
def get_numpy(metric, marginal=False):
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
    X = np.zeros((len(avg_dfs["us-east-1"]), len(name_vector)))
    if marginal:
        X = np.zeros((len(mar_dfs["us-east-1"]), len(name_vector)))

    for i, name in enumerate(name_vector):
        if name == "none":
            continue
        if marginal:
            X[:, i] = mar_dfs[name]["marginal_carbon_avg"].values
        else:
            X[:, i] = avg_dfs[name]["carbon_intensity_avg"].values

    # print the first few rows of the matrix
    # print(X[:5, :])

    # save the datetimes to a separate pandas series
    datetimes = avg_dfs["us-east-1"]["datetime"]
    if marginal:
        datetimes = mar_dfs["us-east-1"]["datetime"]

    # print(datetimes.head())

    return datetimes, X

def get_simplex(simplex_names, marginal=False, forecast=False):
    # get the "column names" of the vectors for the tree embedding in the metric space
    name_vector = simplex_names.copy()
    marginal_target = "marginal_forecast_avg" if forecast else "marginal_carbon_avg"

    for i, name in enumerate(name_vector):
        if "OFF" in name:
            name_vector[i] = "none"
        if "ON" in name:
            name_vector[i] = name.replace(" ON", "")

    # create a numpy matrix where each column is the corresponding carbon intensity trace for that region
    X = np.zeros((len(avg_dfs["us-east-1"]), len(name_vector)))
    if marginal or forecast:
        X = np.zeros((len(mar_dfs["us-east-1"]), len(name_vector)))

    for i, name in enumerate(name_vector):
        if name == "none":
            continue
        if marginal or forecast:
            X[:, i] = mar_dfs[name][marginal_target].values
        else:
            X[:, i] = avg_dfs[name]["carbon_intensity_avg"].values

    # save the datetimes to a separate pandas series
    datetimes = avg_dfs["us-east-1"]["datetime"]
    if marginal or forecast:
        datetimes = mar_dfs["us-east-1"]["datetime"]

    return X


