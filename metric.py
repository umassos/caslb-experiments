# metric space from data set, FRT tree embedding implementations
# Adam Lechowicz
# Jul 2024

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
from collections import defaultdict

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

class Node:
    def __init__(self, point):
        self.point = point
        self.children = []

class Tree:
    def __init__(self, root):
        self.root = root


class MetricSpace:
    def __init__(self, tau=1, names=names):
        self.names = names
        self.tau = tau
        self.df = pd.read_csv("latencies.csv", index_col=0)
        self.avg_distances, self.std_devs = self.get_distances(self.df)
        self.simplex_names, self.c_simplex, self.simplex_distances = self.generate_simplex_distances()

        # get the tree embedding
        self.tree, self.weights, self.levels = self.frt_algorithm(self.names)
        self.weight_vector = self.get_weight_vector()
        self.unit_c_vector, self.name_vector = self.get_unit_c_vector()
        self.phi_inv = self.phi_inverse(self.names, self.name_vector, self.simplex_names)
        self.phi_mat = self.phi(self.names, self.name_vector, self.simplex_names)


    # change this for subset
    def get_names(self):
        return self.names
    
    def refresh_tree(self):
        self.tree, self.weights, self.levels = self.frt_algorithm(self.names)
        self.weight_vector = self.get_weight_vector()
        self.unit_c_vector, self.name_vector = self.get_unit_c_vector()
        self.phi_inv = self.phi_inverse(self.names, self.name_vector, self.simplex_names)
        self.phi_mat = self.phi(self.names, self.name_vector, self.simplex_names)

    # loads data and computes the average and std dev of latencies between the AWS regions
    def get_distances(self, df):
        # compute average and std dev of latencies between the AWS regions and store them in a matrix
        avg_distances = np.zeros((len(self.names), len(self.names)))
        std_dev_distances = np.zeros((len(self.names), len(self.names)))
                            
        for from_region in self.names:
            for to_region in self.names:
                if from_region == to_region:
                    continue
                # filter the dataframe to get the latencies between the two regions
                latencies_from = df[(df['from'] == from_region) & (df['to'] == to_region)]['latency']
                latencies_to = df[(df['from'] == to_region) & (df['to'] == from_region)]['latency']
                # combine the two series
                latencies = pd.concat([latencies_from, latencies_to])
                # compute the average and std dev
                avg_distances[self.names.index(from_region), self.names.index(to_region)] = latencies.mean()
                avg_distances[self.names.index(to_region), self.names.index(from_region)] = latencies.mean()
                std_dev_distances[self.names.index(from_region), self.names.index(to_region)] = latencies.std()
                std_dev_distances[self.names.index(to_region), self.names.index(from_region)] = latencies.std()
        
        return avg_distances, std_dev_distances
    
    # given distance matrix, computes distance between two points
    def distance(self, point1, point2):
        return self.avg_distances[self.names.index(point1), self.names.index(point2)]
    
    # given distance, compute a ball of points around a point
    def ball(self, points, point, radius):
        """
        Return the set of points within the ball of the given radius centered at the given point.
        """
        point_set = set()
        for p in points:
            d = self.distance(point, p)
            if d <= radius:
                point_set.add(p)
        return point_set
    
    # given distance and set of points, compute the diameter
    def diameter(self, points=None):
        if points is None:
            points = self.names
        return max(self.distance(p1, p2) for p1 in points for p2 in points)
    
    # generate the names of states in the probability simplex
    def generate_simplex_distances(self):
        simplex_names = []
        c_simplex = []
        for name in self.names:
            simplex_names.append(name + " ON")
            c_simplex.append(1)
            simplex_names.append(name + " OFF")
            c_simplex.append(0)

        # generate distances
        simplex_distances = np.zeros((len(simplex_names), len(simplex_names)))
        for from_region in simplex_names:
            for to_region in simplex_names:
                if from_region == to_region:
                    continue
                from_name = from_region.split(" ")[0]
                to_name = to_region.split(" ")[0]
                from_state = from_region.split(" ")[1]
                to_state = to_region.split(" ")[1]
                # if the states are different
                if from_name != to_name:
                    if from_state == "ON" and to_state == "ON":
                        simplex_distances[simplex_names.index(from_region), simplex_names.index(to_region)] = self.distance(from_name, to_name)
                    elif from_state == "OFF" and to_state == "OFF":
                        simplex_distances[simplex_names.index(from_region), simplex_names.index(to_region)] = self.distance(from_name, to_name) + 2*self.tau
                    else:
                        simplex_distances[simplex_names.index(from_region), simplex_names.index(to_region)] = self.distance(from_name, to_name) + self.tau
                # if the names are the same
                else:
                    simplex_distances[simplex_names.index(from_region), simplex_names.index(to_region)] = self.tau
        return simplex_names, np.array(c_simplex), simplex_distances


    # generate a new, random distance matrix between states in the probability simplex
    def shuffled_distances(self, numshuffles=5):
        names = self.names
        new_simplex_distances = np.zeros((len(self.simplex_names), len(self.simplex_names)))
        new_name_vector = self.name_vector.copy()
        new_c_simplex = self.c_simplex.copy()
        for i in range(numshuffles):
            # pick a random region to swap
            swap1 = random.choice(names)
            swap2 = random.choice(names)
            while swap1 == swap2:
                swap2 = random.choice(names)
            # swap the distances
            swap1indexON = self.simplex_names.index(swap1 + " ON")
            swap2indexON = self.simplex_names.index(swap2 + " ON")

            for from_region in self.simplex_names:
                from_name = from_region.split(" ")[0]
                from_state = from_region.split(" ")[1]
                from_index = self.simplex_names.index(from_region)
                if from_name == swap1:
                    from_index = self.simplex_names.index(swap2 + " " + from_state)
                elif from_name == swap2:
                    from_index = self.simplex_names.index(swap1 + " " + from_state)
                for to_region in self.simplex_names:
                    if from_region == to_region:
                        continue
                    to_name = to_region.split(" ")[0]
                    to_state = to_region.split(" ")[1]
                    to_index = self.simplex_names.index(to_region)
                    if to_name == swap1:
                        to_index = self.simplex_names.index(swap2 + " " + to_state)
                    elif to_name == swap2:
                        to_index = self.simplex_names.index(swap1 + " " + to_state)
                    # if the states are different
                    if from_name != to_name:
                        new_simplex_distances[from_index, to_index] = self.distance(from_name, to_name)
                    # if the names are the same
                    else:
                        new_simplex_distances[from_index, to_index] = self.tau

            # swap the c values
            new_c_simplex[swap1indexON] = self.c_simplex[swap2indexON]
            new_c_simplex[swap2indexON] = self.c_simplex[swap1indexON]

            # swap the vector names
            for i, name in enumerate(new_name_vector):
                new_name = name.replace(swap1, "INTERMEDIATE").replace(swap2, swap1).replace("INTERMEDIATE", swap2)
                new_name_vector[i] = new_name
        
        new_phi = self.phi(names, new_name_vector, self.simplex_names)

        return new_simplex_distances, new_c_simplex, new_phi



    # generates a random tree embedding of the metric space according to the FRT algorithm
    def frt_algorithm(self, points):
        diam = self.diameter(points)
        n = len(points)
        log_delta = np.ceil(np.log2(diam))

        # permute the points and save them in pi
        pi = np.random.permutation(points)

        # choose r_0
        radius_0 = np.random.uniform(0.5, 1)
        # radius_0 = 1
        radii = [radius_0 * 2**i for i in range(1, int(log_delta) + 1)]

        # set of nodes at each level (dict)
        levels = defaultdict(list)
        ancestors = defaultdict(list)
        edgeweights = defaultdict(float)
        levels[log_delta] = [frozenset(points)]
        ancestors[log_delta] = None
        end = 1

        for i in reversed(range(1, int(log_delta)+1)):
            # get the sets of nodes at level i
            Cs = levels[i]
            # if the length of Cs is n, we are done
            if len(Cs) == n:
                end = i
                break
            # print("i: {}, Cs: {}".format(i,Cs))
            for C in Cs:
                # print("i: {}, C: {}".format(i,C))
                S = C.copy()
                for j in range(0, n):
                    B = self.ball(points, pi[j], radii[i-1])
                    # print("i: {}, C: {}, radius: {}, B: {}".format(i, C, radii[i-1], B))
                    P = S.intersection(B)
                    # if P is not empty...
                    if len(P) > 0:
                        S = S.difference(P)
                        # add P to T as a child of C at level i-1
                        levels[i-1].append(frozenset(P))
                        ancestors[i-1].append(frozenset(C))
            edgeweights[i-1] = radii[i-1]

        # build the tree
        root = Node(frozenset(points))
        tree = Tree(root)
        cur_level = [root]
        completed = set()
        for i in reversed(range(end, int(log_delta))):
            next_level = []
            for C, parent in zip(levels[i], ancestors[i]):
                # get parent node from current level
                parent_node = None
                for node in cur_level:
                    if node.point == parent:
                        parent_node = node
                        break
                # check if the parent and C are identical with length 1
                if parent == C and len(C) == 1:
                    continue
                # create a new node
                new_node = Node(C)
                parent_node.children.append(new_node)
                next_level.append(new_node)
                # if C has length one, add it to the completed set and add an OFF state
                if len(C) == 1:
                    completed.add(C)
                    leaf_node = Node("OFF")
                    new_node.children.append(leaf_node)
            cur_level = next_level
        
        return tree, edgeweights, log_delta
    
    # get the weight vector for the tree
    def get_weight_vector(self):
        weight_vector = [0]
        # iterate through the tree using breadth first search, each time we encounter a node, we add the weight of the preceding edge to the weight vector
        queue = [self.tree.root]
        level_queue = [self.levels]
        while queue:
            node = queue.pop(0)
            level = level_queue.pop(0)
            for child in node.children:
                if len(node.children) == 1 and child.point == "OFF":
                    queue.append(child)
                    level_queue.append(level-1)
                    weight_vector.append(self.tau)
                else:
                    queue.append(child)
                    level_queue.append(level-1)
                    weight_vector.append(self.weights[level-1])
        return np.array(weight_vector)
    
    # get the unit_c_vector for the tree and the name vector
    def get_unit_c_vector(self):
        c_vector = [0]
        name_vector = ["root"]
        # iterate through the tree using breadth first search, each time we encounter a node, we add a 1 to the c vector if it is an ON node, otherwise we add a 0
        queue = [self.tree.root]
        level_queue = [self.levels]
        while queue:
            node = queue.pop(0)
            level = level_queue.pop(0)
            for child in node.children:
                queue.append(child)
                level_queue.append(level-1)
                if len(child.point) == 1:
                    c_vector.append(1)
                    name_vector.append(list(child.point)[0])
                else:
                    c_vector.append(0)
                    string = str(list(node.point))
                    if child.point == "OFF":
                        string = list(node.point)[0] + " OFF"
                    name_vector.append(string)
        return np.array(c_vector), name_vector
    
    # define a phi (inverse) matrix that maps regular vectors to simplex vectors
    def phi_inverse(self, names, vector_names, simplex_names):
        phi_inv_matrix = np.zeros((len(simplex_names), len(vector_names)))
        for i, name in enumerate(names):
            # at the simplex_names[name + " 0N"] row, we have one 1 at the name + " ON" column and one -1 at the name + " OFF" column
            phi_inv_matrix[simplex_names.index(name + " ON"), vector_names.index(name)] = 1
            phi_inv_matrix[simplex_names.index(name + " ON"), vector_names.index(name + " OFF")] = -1
            # at the simplex_names[name + " OFF"] row, we have one 1 at the name + " OFF" column
            phi_inv_matrix[simplex_names.index(name + " OFF"), vector_names.index(name + " OFF")] = 1
        return phi_inv_matrix
    
    # define a phi matrix that maps simplex vectors to regular vectors
    def phi(self, names, vector_names, simplex_names):
        phi_matrix = np.zeros((len(vector_names), len(simplex_names)))
        for i, name in enumerate(names):
            # at the vector_names[name] row, we have one 1 at the simplex_names[name + " ON"] row, and one 1 at the simplex_names[name + " OFF"] row
            phi_matrix[vector_names.index(name), simplex_names.index(name + " ON")] = 1
            phi_matrix[vector_names.index(name), simplex_names.index(name + " OFF")] = 1
            # at the vector_names[name + " OFF"] row, we have one 1 at the simplex_names[name + " OFF"] row
            phi_matrix[vector_names.index(name + " OFF"), simplex_names.index(name + " OFF")] = 1
            # at ANY OTHER row where the name is present, we have one 1 at the simplex_names[name + " ON"] row and one 1 at the simplex_names[name + " OFF"] row
            for j, vector_name in enumerate(vector_names):
                if j in [vector_names.index(name), vector_names.index(name + " OFF")]:
                    continue
                if name in vector_name:
                    phi_matrix[j, simplex_names.index(name + " ON")] = 1
                    phi_matrix[j, simplex_names.index(name + " OFF")] = 1
        # at the vector_names["root"] row, we have all 1s
        phi_matrix[0, :] = 1
        return phi_matrix
    
    # convert ANY regular vector into a simplex vector (Phi inverse)
    def convert_to_simplex(self, vector):
        simplex_vector = np.zeros(len(self.simplex_names))
        for name in self.names:
            simplex_vector[self.simplex_names.index(name + " ON")] = vector[self.name_vector.index(name)] - vector[self.name_vector.index(name + " OFF")]
            simplex_vector[self.simplex_names.index(name + " OFF")] = vector[self.name_vector.index(name + " OFF")]
        return simplex_vector
    
    # convert a simplex vector into a regular vector (Phi)
    def convert_to_regular(self, simplex_vector):
        vector = np.zeros(len(self.name_vector))
        vector[0] = 1
        for name in self.names:
            name_index = self.name_vector.index(name)
            name_OFF_index = self.name_vector.index(name + " OFF")  
            cumulative = simplex_vector[self.simplex_names.index(name + " ON")] + simplex_vector[self.simplex_names.index(name + " OFF")]
            
            vector[name_index] = cumulative
            vector[name_OFF_index] = simplex_vector[self.simplex_names.index(name + " OFF")]
            for i, vector_name in enumerate(self.name_vector):
                if i in [name_index, name_OFF_index]:
                    continue
                if name in vector_name:
                    vector[i] += cumulative
        return vector
    
    def get_start_state(self, name):
        vector = np.zeros(len(self.name_vector))
        simplex_vector = np.zeros(len(self.simplex_names))
        simplex_vector[self.simplex_names.index(name + " OFF")] = 1
        vector = self.convert_to_regular(simplex_vector)
        return vector, simplex_vector

        
    # pretty prints the current tree embedding to the console
    def print_tree(self):
        self.print_tree_helper(self.tree.root)

    # helper to print the tree
    def print_tree_helper(self, node, level=0):
        if node:
            print(' ' * level * 10, str(list(node.point)))
            for child in node.children:
                self.print_tree_helper(child, level + 1)
    

    

