import csv
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath):
    data = [] 
    with open(filepath, mode='r', encoding='utf-8') as file:
        file = csv.DictReader(file)
        for row in file:
            data.append(dict(row))
    return data

def calc_features(row):
    features_list = list(row.values())[2:]
    features_array = np.array(features_list, dtype=np.float64)
    return features_array

def hac(features):
    n = len(features)
    Z = np.zeros((n - 1, 4))
    cluster_map = list(range(n))
    cluster_sizes = [1] * n  

    # DISTANCE MATRIX CREATION
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i, j] = distance_matrix[j, i] = np.linalg.norm(features[i] - features[j])
    np.fill_diagonal(distance_matrix, np.inf)

    for k in range(n - 1):
        # GET CLOSEST CLUSTERS 
        distance_matrix_nodiag = distance_matrix[distance_matrix != np.inf]
        min_value = np.min(distance_matrix_nodiag)
        rows, cols = np.where(distance_matrix == min_value)
        pairs = []
        for l in range(len(rows)):
            pair = (min(rows[l], cols[l]), max(rows[l], cols[l]))
            pairs.append(pair)
        row, col = pairs[0]
        if row > col:
            row, col = col, row

        # STORE Z VALUES
        z0, z1 = sorted([cluster_map[row], cluster_map[col]])
        z2 = min_value
        z3 = cluster_sizes[row] + cluster_sizes[col]

        Z[k,0] = z0
        Z[k,1] = z1
        Z[k,2] = z2
        Z[k,3] = z3


        # UPDATE CLUSTER MAP AND SIZES
        new_index = n + k
        cluster_map[row] = new_index
        cluster_sizes[row] = Z[k, 3] 
        cluster_map[col] = -1 #removal
        cluster_sizes[col] = 0  

        # UPDATE DISTANCE MATRIX 
        for i in range(n):
            if i != row and cluster_map[i] != -1:
                d = max(distance_matrix[row, i], distance_matrix[col, i])
                distance_matrix[row, i] = d
                distance_matrix[i, row] = d
        distance_matrix[:, col] = np.inf 
        distance_matrix[col, :] = np.inf

        distance_matrix[row, row] = np.inf

    return Z

def fig_hac(Z,names):
    fig = plt.figure()
    dendrogram(Z, labels = names, leaf_rotation = 90)
    plt.tight_layout()
    return fig

def normalize_features(features):
    features_array = np.array(features)
    means = np.mean(features_array, axis=0)
    stds = np.std(features_array, axis=0)
    norm_features = (features_array - means) / stds
    norm_features_list = list(map(np.array, norm_features))
    
    return norm_features_list
