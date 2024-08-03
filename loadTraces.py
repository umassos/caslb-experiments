# Data Loading Helper (loads cloud traces in the form of MATLAB .mat files)

from mat4py import loadmat
import numpy as np
from math import e
import itertools

# load cloud traces
traceLengths = []
for i in range(1, 87):
    # loads a single trace from a single MAT file
    data = loadmat('cloud_job_lengths/jobtrace{}.mat'.format(i))
    lengths = data['jobvalueCell']
    
    for x in lengths:
        traceLengths.append(x[0])

traceLengths = np.array(traceLengths)


# generate a random (integer) job length according to the distribution of the traces
def randomJobLength(down, up):
    # normalize the trace lengths to fall between up and down
    normLengths = np.array(traceLengths)
    normLengths = normLengths - np.min(normLengths)
    normLengths = normLengths / np.max(normLengths)
    normLengths = normLengths * (up - down)
    normLengths = normLengths + down
    # sample a random job length
    choice = np.random.choice(normLengths)
    return int(choice)

def returnTraceLengths():
    return traceLengths