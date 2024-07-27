import pandas as pd
from sklearn import metrics
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score 
from scipy.stats import kendalltau
from scipy.linalg import det
from scipy.stats import pointbiserialr
from itertools import combinations
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans  # Replace with your preferred clustering algorithm
import PySimpleGUI as sg
from sklearn.preprocessing import MinMaxScaler
import time

import numpy as np

def calculate_ball_hall_manual(data, labels): 
    num_clusters = len(np.unique(labels))
    cluster_centers = [np.mean(data[labels == i], axis=0) for i in range(num_clusters)]
 
    within_cluster_distances = []

    for i in range(num_clusters):
        cluster_points = data[labels == i]
        distance_sum = np.sum(np.linalg.norm(cluster_points - cluster_centers[i], axis=1))
        within_cluster_distances.append(distance_sum / len(cluster_points))

    ball_hall_index = np.mean(within_cluster_distances)
    return ball_hall_index

def calculate_banfeld_raftery_manual(data, labels):
    num_clusters = len(np.unique(labels))
    cluster_sizes = [len(labels[labels == i]) for i in range(num_clusters)]
    total_points = len(labels) 
 
    banfeld_raftery_index = 0 

    for i in range(num_clusters):
        cluster_points = data[labels == i]
        cluster_size = cluster_sizes[i]
        distance_sum = np.sum(np.linalg.norm(cluster_points - np.mean(cluster_points, axis=0), axis=1))
        banfeld_raftery_index += (distance_sum / cluster_size) * (cluster_size / total_points)

    return banfeld_raftery_index

def calculate_cindex_manual(data, labels):
    num_samples, num_features = data.shape
    concordant_pairs = 0
    total_pairs = 0

    for i in range(num_samples - 1):
        for j in range(i + 1, num_samples):
            # Check if the pair is comparable
            if labels[i] != labels[j]:
                total_pairs += 1

                # Check for concordance
                try:
                    coef = all(data[i, k] < data[j, k] and labels[i] < labels[j] or
                               data[i, k] > data[j, k] and labels[i] > labels[j]
                               for k in range(num_features))
                    
                    if coef:
                        concordant_pairs += 1
                except Exception as e:
                    print(f"Error: {e}")
                    print("i:", i)
                    print("j:", j)
                    print("data:")
                    print(data)
                    print("labels[i]:", labels[i])
                    print("labels[j]:", labels[j])

    cindex = concordant_pairs / total_pairs if total_pairs > 0 else 0
    return cindex

def calculate_calinski_harabasz_manual(data, labels):
    num_clusters = len(set(labels))
    num_samples = len(data)
    
    overall_mean = np.mean(data, axis=0)
    
 ##   within_cluster_variance_mean = 0
    
    within_cluster_variance = 0
    for i in range(num_clusters):
        cluster_points = data[labels == i]
        cluster_mean = np.mean(cluster_points, axis=0)
        within_cluster_variance += np.sum((cluster_points - cluster_mean) ** 2)
    
    between_cluster_variance = 0
    for i in range(num_clusters):
        cluster_points = data[labels == i]
        cluster_mean = np.mean(cluster_points, axis=0)
        between_cluster_variance += len(cluster_points) * np.sum((cluster_mean - overall_mean) ** 2)
    
    calinski_harabasz = (between_cluster_variance / (num_clusters - 1)) / (within_cluster_variance / (num_samples - num_clusters))
    
    print("between_cluster_variance: ")
    print(between_cluster_variance)
    
    print("within_cluster_variance: ")
    print(within_cluster_variance)
    
    print("num_clusters: ")
    print(num_clusters)
    
    print("num_samples: ")
    print(num_samples)
    
    print("calinski_harabasz: ")
    print(calinski_harabasz)
    
    print("printed")
    
    # import sys
    # sys.exit()
    
    return calinski_harabasz

def calculate_davies_bouldin_manual(data, labels):
    num_clusters = len(set(labels))
    
    cluster_centers = np.array([np.mean(data[labels == i], axis=0) for i in range(num_clusters)])
    
    cluster_distances = pairwise_distances(cluster_centers, metric='euclidean')
    
    avg_davies_bouldin = 0
    
    for i in range(num_clusters):
        other_clusters = [j for j in range(num_clusters) if j != i]
        similarity = (cluster_distances[i, other_clusters] + cluster_distances[other_clusters, i]) / 2
        avg_davies_bouldin += np.max(similarity)
    
    davies_bouldin_index = avg_davies_bouldin / num_clusters
    
    return davies_bouldin_index

def calculate_dunn_index_manual(data, labels):
    num_clusters = len(set(labels))
    
    cluster_centers = np.array([np.mean(data[labels == i], axis=0) for i in range(num_clusters)])
    
    max_intra_cluster_distance = 0
    min_inter_cluster_distance = np.inf
    
    for i in range(num_clusters):
        intra_cluster_distances = pairwise_distances(data[labels == i], metric='euclidean')
        avg_intra_cluster_distance = np.mean(intra_cluster_distances)
        
        if avg_intra_cluster_distance > max_intra_cluster_distance:
            max_intra_cluster_distance = avg_intra_cluster_distance
        
        for j in range(i + 1, num_clusters):
            inter_cluster_distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            if inter_cluster_distance < min_inter_cluster_distance:
                min_inter_cluster_distance = inter_cluster_distance
    
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance
    
    return dunn_index

def calculate_baker_hubert_gamma(data, labels):
    num_clusters = len(set(labels))
    
    cluster_centers = np.array([np.mean(data[labels == i], axis=0) for i in range(num_clusters)])
    
    intra_cluster_distances = []
    inter_cluster_distances = []
    
    for i in range(num_clusters):
        intra_cluster_distances.extend(pairwise_distances(data[labels == i], metric='euclidean').flatten())
        
        for j in range(i + 1, num_clusters):
            inter_cluster_distances.append(np.linalg.norm(cluster_centers[i] - cluster_centers[j]))
    
    avg_intra_cluster_distance = np.mean(intra_cluster_distances)
    avg_inter_cluster_distance = np.mean(inter_cluster_distances)
    
    baker_hubert_gamma = avg_intra_cluster_distance / avg_inter_cluster_distance
    
    return baker_hubert_gamma

def calculate_gdi(data, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Calculate cluster diameters
    cluster_diameters = [np.max(pairwise_distances(data[labels == i])) for i in unique_labels]

    # Calculate inter-cluster distances
    inter_cluster_distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            distance = np.min(pairwise_distances(data[labels == i], data[labels == j]))
            inter_cluster_distances.append(distance)

    # Calculate GDI
    min_inter_cluster_distance = np.min(inter_cluster_distances)
    max_intra_cluster_diameter = np.max(cluster_diameters)

    if max_intra_cluster_diameter == 0:
        # Avoid division by zero if there's a cluster with only one point
        gdi_index = 0
    else:
        gdi_index = min_inter_cluster_distance / max_intra_cluster_diameter

    return gdi_index

def calculate_g_plus(data, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Calculate cluster diameters
    cluster_diameters = [np.max(pairwise_distances(data[labels == i])) for i in unique_labels]

    # Calculate inter-cluster distances
    inter_cluster_distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            distance = np.min(pairwise_distances(data[labels == i], data[labels == j]))
            inter_cluster_distances.append(distance)

    # Calculate G_plus
    sum_inter_cluster_distances = np.sum(inter_cluster_distances)
    sum_intra_cluster_diameters = np.sum(cluster_diameters)

    if sum_intra_cluster_diameters == 0:
        # Avoid division by zero if there's a cluster with only one point
        g_plus_index = 0
    else:
        g_plus_index = sum_inter_cluster_distances / sum_intra_cluster_diameters

    return g_plus_index

def calculate_ksqdetw(data, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Calculate Kendall's tau distance within clusters
    kendall_tau_distances = []
    for label in unique_labels:
        cluster_data = data[labels == label]
        cluster_size = len(cluster_data)
        
        if cluster_size > 1:
            # Calculate Kendall's tau distance matrix
            kendall_tau_matrix = np.zeros((cluster_size, cluster_size))
            
            cluster_data_list = cluster_data.drop('x', axis=1).values.tolist()
            cluster_data = cluster_data_list
            
            print("Shape: ( " + str(len(cluster_data)) + " , " + str(len(cluster_data[0])) + " )")
            
            # Calculate pairwise Kendall's tau distances for each feature
            for i, j in combinations(range(cluster_size), 2):
                for feature in range(len(cluster_data[0])):
                    print("i: " + str(i)) 
                    print(type(i))
                    print("feature: " + str(feature))
                    print(type(feature))
                    print("cluster data:")
                    print(cluster_data)
                    print("cluster info: ")
                    print(cluster_data[i][feature]) 
                    c = kendalltau(cluster_data[i][feature], cluster_data[j][feature], method='asymptotic')
                    kendall_tau, _ = c  
                    kendall_tau_matrix[i, j] += kendall_tau
                    kendall_tau_matrix[j, i] += kendall_tau
             
            # Normalize by the number of features
            kendall_tau_matrix /= data.shape[1]  
            
            # Calculate average Kendall's tau distance within the cluster
            avg_kendall_tau_distance = np.sum(kendall_tau_matrix) / (cluster_size * (cluster_size - 1))
            kendall_tau_distances.append(avg_kendall_tau_distance)

    # Calculate KsqDetW
    ksqdetw_index = np.sum(np.square(kendall_tau_distances)) / n_clusters
 
    return ksqdetw_index
 
def calculate_log_det_ratio(data, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Calculate Log_Det_Ratio for each cluster
    log_det_ratios = []
    for label in unique_labels:
        cluster_data = data[labels == label]

        # Calculate covariance matrix for the cluster
        covariance_matrix = np.cov(cluster_data, rowvar=False)

        # Calculate determinant of the covariance matrix
        det_covariance_matrix = det(covariance_matrix)

        log_det_ratios.append(np.log(det_covariance_matrix))

    # Calculate Log_Det_Ratio index
    log_det_ratio_index = np.sum(log_det_ratios) / n_clusters

    return log_det_ratio_index

def calculate_log_trace_ratio(data, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Calculate mean of the entire dataset
    mean_data = np.mean(data, axis=0)

    # Initialize scatter matrices
    within_scatter_matrix = np.zeros((data.shape[1], data.shape[1]))
    between_scatter_matrix = np.zeros((data.shape[1], data.shape[1]))

    # Calculate scatter matrices
    for label in unique_labels:
        cluster_data = data[labels == label]
        mean_cluster = np.mean(cluster_data, axis=0)
        n_samples_in_cluster = len(cluster_data)

        within_scatter_matrix += np.cov(cluster_data, rowvar=False) * (n_samples_in_cluster - 1)

        between_scatter_matrix += np.outer((mean_cluster - mean_data), (mean_cluster - mean_data)) * n_samples_in_cluster

    # Calculate trace of scatter matrices
    trace_within = np.trace(within_scatter_matrix)
    trace_between = np.trace(between_scatter_matrix)

    # Calculate Log Trace Ratio
    log_trace_ratio = np.log(trace_between / trace_within)

    return log_trace_ratio

def calculate_mcclain_rao(data, labels, metric='braycurtis'):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Calculate pairwise distances for each cluster
    pairwise_distances = []
    for label in unique_labels:
        cluster_data = data[labels == label]
        distances = cdist(cluster_data, cluster_data, metric=metric)
        pairwise_distances.extend(distances[np.triu_indices(len(cluster_data), k=1)])

    # Calculate mean of pairwise distances
    mcclain_rao = np.mean(pairwise_distances)

    return mcclain_rao

def point_biserial_correlation(x, y):
    mean_1 = np.mean(x[y == 1])
    mean_0 = np.mean(x[y == 0])
    
    n_1 = np.sum(y == 1)
    n_0 = np.sum(y == 0)
    n = len(y)
    
    pooled_std = np.sqrt(((n_0 - 1) * np.var(x[y == 0]) + (n_1 - 1) * np.var(x[y == 1])) / (n - 2))
    
    r_pb = (mean_1 - mean_0) / pooled_std * np.sqrt(n_0 * n_1 / (n * (n - 1)))
    
    return r_pb

def calculate_point_bisserial(data, labels):
    
    num_samples, num_features = data.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    data_list = data.drop('x', axis=1).values.tolist()
    data = data_list
    
    # Calculate point-biserial correlations for each pair of points
    pbm_values = [] 
    
 #   for feature_index in range(1, num_features):  # Start from 1 since 'x' is dropped
            
    for i in range(num_samples):
            for j in range(i + 1, num_samples):
                if labels[i] == labels[j]:
                    pbm_values.append(point_biserial_correlation(data[i][:], data[j][:]))
                else:
                    pbm_values.append(-point_biserial_correlation(data[i][:], data[j][:]))
    
    # Calculate the mean of point-biserial correlations
    point_bisserial = np.mean(pbm_values)
    
    return point_bisserial

def calculate_pbm_index(data, labels):
    """
    Calculate the Partition-Based Metric (PBM) index.

    Parameters:
    - data: List or array of data points.
    - labels: List or array of cluster labels corresponding to each data point.

    Returns:
    - PBM index value (float).
    """

    unique_labels = list(set(labels))
    clusters = {label: [i for i, x in enumerate(labels) if x == label] for label in unique_labels}

    # Convert data points to sets for faster intersection calculations
    data_sets = [set(cluster) for cluster in clusters.values()]

    # Calculate numerator and denominator
    numerator = sum(len(cluster1 & cluster2) for cluster1, cluster2 in combinations(data_sets, 2))
    denominator = sum(len(cluster) for cluster in data_sets)

    # Avoid division by zero
    if denominator == 0:
        return 0.0

    # Calculate PBM index
    pbm_index = numerator / denominator

    return pbm_index

def ratkowski_lance_index(data, labels):
    """
    Compute the Ratkowsky Lance Index (RKI) for the given data and labels.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.

    Returns:
    - rki: The Ratkowsky Lance Index value.
    """

    # Number of dimensions (columns)
    p = data.shape[1] - 1

    # Compute the mean R of the quotients between BGSS and TSS for each dimension    
    r_values = []
    for j in range(p):
        bgss_j = 0
        for k in np.unique(labels):
            # Extract data points for the current cluster
            # cluster_data = data[True, j]
            # print("cluster data example: ")
            # print(cluster_data) 
            
            x = np.where(labels == k)[0].tolist()
            
            print("x: ")
            print(x)
            
            cluster_data = []
            
    #        data_list = data[1:]       

            if not isinstance(data, list):   
                data_list = data.drop('x', axis=1).values.tolist()
                data = data_list            
            
            print("data: ") 
 #           print(data.shape)  
            print(type(data))
            
            for i in x:
                print("[ " + str(i) + " , " + str(j) + " ] ")
                cl_data = data[i][j]
                cluster_data.append(cl_data)
            
            if len(cluster_data) > 0:
                # Compute BGSSj for the current cluster
                bgss_j += len(cluster_data) * np.var(cluster_data)
    
        # Compute TSSj
        tss_j = np.var(data[:][j])
    
        # Compute R for each dimension
        r_values.append(bgss_j / tss_j)

    # Compute the mean R
    mean_r = np.mean(r_values)

    # Compute the Ratkowsky Lance Index (RKI)
    rki = mean_r / p

    return rki
	
def ray_turi_index(data, labels):
    """
    Compute the Ray-Turi Index for the given data and labels.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.

    Returns:
    - ray_turi: The Ray-Turi Index value.
    """

    # Number of data points
    N = len(data)

    # Number of features
    num_features = data.shape[1]

    # Number of clusters
    K = len(np.unique(labels))

    # Calculate the barycenters of each cluster
    cluster_barycenters = [np.mean(data[labels == k], axis=0) for k in range(1, K + 1)]

    if not isinstance(data, list):   
        data_list = data.drop('x', axis=1).values.tolist()
        data = data_list 
        
    print("data: ")
    print(data)

    numerator = 0
    # Calculate the numerator (WGSS - within-group sum of squares)
    for i in range(N):
        for j in range(num_features-1):
            print("i: " + str(i))
            print("j: " + str(j))
            di = data[i][j] - cluster_barycenters[labels[i] - 1][j]
            numerator += np.linalg.norm(di) ** 2

    # Calculate the denominator (minimum squared distances between cluster barycenters)
    combinations_barycenters = list(combinations(cluster_barycenters, 2))
    denominator = min(np.linalg.norm(bc1 - bc2) ** 2 for bc1, bc2 in combinations_barycenters)

    # Compute the Ray-Turi Index
    ray_turi = numerator / (N * denominator)

    return ray_turi 
 
def scott_symons_index(data, labels):
    """
    Compute the Scott-Symons Index for the given data and labels.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.

    Returns:
    - scott_symons: The Scott-Symons Index value.
    """

    # Number of data points
    N = len(data)

    # Number of clusters
    K = len(np.unique(labels))

    # Initialize Scott-Symons Index
    scott_symons = 0.0

    for k in range(1, K + 1):
        # Extract data points in cluster k
        cluster_data = data[labels == k]

        # Number of data points in cluster k
        nk = len(cluster_data)

        # Compute the covariance matrix of cluster k
        covariance_matrix = np.cov(cluster_data, rowvar=False)

        # Compute the determinant of the covariance matrix, avoiding potential numerical issues
        log_det = np.linalg.slogdet(covariance_matrix)[1]

        # Update Scott-Symons Index
        scott_symons += nk * log_det

    # Normalize by the total number of data points
    scott_symons /= N

    return scott_symons
	
def calculate_SD_index(data, labels):
    """
    Compute the SD (Scattering and Separation Distance) Index for the given data and labels.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.

    Returns:
    - SD_index: The SD Index value.
    """
    # Number of data points
    N = len(data)

    # Number of clusters
    K = len(np.unique(labels))

    # Calculate the average scattering (S)
    V = np.var(data, axis=0)
    scattering = 0  # Initialize scattering

    for k in range(1, K + 1):
        fp = np.var(data[labels == k], axis=0)
        scattering += np.mean(np.linalg.norm(fp / np.linalg.norm(V)))

    # Calculate the total separation (D)
    barycenters = [np.mean(data[labels == k], axis=0).values for k in range(1, K + 1)]

    D_max = max(np.linalg.norm(np.array(bc1) - np.array(bc2)) for bc1 in barycenters for bc2 in barycenters if not np.array_equal(np.array(bc1), np.array(bc2)))
    D_min = min(np.linalg.norm(np.array(bc1) - np.array(bc2)) for bc1 in barycenters for bc2 in barycenters if not np.array_equal(np.array(bc1), np.array(bc2)))
    separation = D_max / D_min

    # Calculate the SD Index
    SD_index = scattering + separation

    return SD_index
    
def calculate_cohesion(data, labels):
    """
    Calculate the cohesion for the given data and labels.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.

    Returns:
    - cohesion: The cohesion value.
    """
    N = len(data)
    cohesion = 0.0

    for k in np.unique(labels):
        cluster_points = data[labels == k]
        pairwise_distances = cdist(cluster_points, cluster_points)
        cohesion += np.sum(pairwise_distances)

    return cohesion / N

def calculate_separation(data, labels):
    """
    Calculate the separation for the given data and labels.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.

    Returns:
    - separation: The separation value.
    """
    K = len(np.unique(labels))
    centroids = [np.mean(data[labels == k], axis=0) for k in range(1, K + 1)]
    centroid_avg = np.mean(data, axis=0)

    separation = np.sum([np.linalg.norm(centroid - centroid_avg) for centroid in centroids]) / K

    return separation

def calculate_standard_deviation(cohesion_values):
    """
    Calculate the standard deviation for the given cohesion values.

    Parameters:
    - cohesion_values: A 1D numpy array or list containing cohesion values for each cluster.

    Returns:
    - standard_deviation: The standard deviation value.
    """
    K = len(cohesion_values)
    cohesion_avg = np.mean(cohesion_values)

    standard_deviation = np.sqrt(np.sum((cohesion_values - cohesion_avg) ** 2) / K)

    return standard_deviation

def calculate_sdbw(data, labels, epsilon=1e-6):
    """
    Calculate the SDbw (Silhouette, Davies-Bouldin, and within-cluster dispersion) index.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.
    - epsilon: A small value to avoid division by zero.

    Returns:
    - sdbw: The SDbw index value.
    """
    cohesion = calculate_cohesion(data, labels)
    separation = calculate_separation(data, labels)

    cohesion_values = [calculate_cohesion(data[labels == k], labels[labels == k]) for k in np.unique(labels)]
    standard_deviation = calculate_standard_deviation(cohesion_values)

    sdbw = cohesion / (separation + epsilon * standard_deviation)

    return sdbw

def calculate_tau_index(data, labels):
    """
    Calculate the Tau index for clustering.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.

    Returns:
    - tau_index: The Tau index value.
    """
    print("Shape: ")
    print(data.shape)
    (N, N2) = data.shape   
    K = len(np.unique(labels))

    # Create a vector B containing values 0 and 1 for between-cluster and within-cluster pairs
    vector_B = np.zeros((N, N2))
    for i in range(N):
        for j in range(N2):
            if labels[i] != labels[j]:
                vector_B[i, j] = 1

    vector_B_flat = vector_B.flatten() 

    # Calculate Kendall Tau correlation
    tau, _ = kendalltau(data.values.flatten(), vector_B_flat)

    # Calculate the number of between-cluster and within-cluster pairs
    NB = np.sum(vector_B_flat == 1)
    NW = N * (N - 1) // 2 - NB 

    # Calculate the Tau index
    tau_index = tau / (NB * NW / (N * (N - 1) / 2))

    return tau_index
    
def calculate_silhouette_index(data, labels):
    """
    Calculate the Silhouette Index for clustering.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.

    Returns:
    - silhouette_index: The Silhouette Index value.
    """
    silhouette_values = silhouette_samples(data, labels)
    silhouette_avg = silhouette_score(data, labels)

    silhouette_index = np.mean(silhouette_values) / silhouette_avg

    return silhouette_index

def calculate_trace_w(data, labels):
    """
    Calculate the Trace W index for clustering.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.

    Returns:
    - trace_w: The Trace W index value.
    """
    K = len(np.unique(labels))
    N = len(data)

    # Calculate the within-group covariance matrix (WG)
    wg = np.zeros((data.shape[1], data.shape[1]))

    for k in range(1, K + 1):
        cluster_data = data[labels == k]
        
        if len(cluster_data) > 0:
        
            print("cluster_data: ")
            print(cluster_data)      
            
            cluster_mean = np.mean(cluster_data, axis=0)       
            
            wg = 0
            
            for point in cluster_data.itertuples(index=False):
                
                print("point: ")
                print(point)
                
                print("cluster_mean: ")
                print(cluster_mean)
                
                x = np.array(point) - np.array(cluster_mean)
                wg += np.outer(x, x)
            
            print("wg: ")
            print(wg)
        
        # for point in cluster_data:
        #     x = point-cluster_mean
        #     wg += np.outer(x, x)

    # Calculate the Trace W index
    trace_w = np.trace(wg) 

    return trace_w

def calculate_trace_wib(data, labels):
    """
    Calculate the Trace WiB (or Trace W 1B) index for clustering.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.

    Returns:
    - trace_wib: The Trace WiB index value.
    """
    K = len(np.unique(labels))
    N = len(data)

    # Calculate the within-group covariance matrix (WG)
    wg = np.zeros((data.shape[1], data.shape[1]))

    for k in range(1, K + 1):
        cluster_data = data[labels == k]
        
        if len(cluster_data) > 0:
        
            print("cluster_data: ")
            print(cluster_data)      
            
            cluster_mean = np.mean(cluster_data, axis=0)       
            
            wg = 0
            
            for point in cluster_data.itertuples(index=False):
                
                print("point: ")
                print(point)
                
                print("cluster_mean: ")
                print(cluster_mean)
                
                x = np.array(point) - np.array(cluster_mean)
                wg += np.outer(x, x)
            
            print("wg: ")
            print(wg)
            
    # Calculate the between-group covariance matrix (BG)
    bg = np.zeros((data.shape[1], data.shape[1]))

    overall_mean = np.mean(data, axis=0)

    for k in range(1, K + 1):
        cluster_data = data[labels == k]
        cluster_mean = np.mean(cluster_data, axis=0)
        bg += len(cluster_data) * np.outer(cluster_mean - overall_mean, cluster_mean - overall_mean)

    # Calculate the Trace WiB index
    
    # Check if WG is singular
    det_wg = np.linalg.det(wg)
    if np.isclose(det_wg, 0.0):
        # Matrix is singular, add regularization
        epsilon = 1e-6
        regularized_wg = wg + epsilon * np.identity(wg.shape[0])
        inverse_wg = np.linalg.inv(regularized_wg)
    else:
        # Matrix is not singular, proceed with inversion
        inverse_wg = np.linalg.inv(wg)
        
    product_result = np.dot(inverse_wg, bg)
    trace_wib = np.trace(product_result)

    return trace_wib
    
def calculate_wemmert_gancarski_index(data, labels):
    """
    Calculate the Wemmert-Gancarski Index for clustering.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.

    Returns:
    - wemmert_gancarski_index: The Wemmert-Gancarski Index value.
    """
    N = len(data)
    K = len(np.unique(labels))

    wemmert_gancarski_values = []

    for k in range(1, K + 1):
        cluster_data = data[labels == k]
        cluster_mean = np.mean(cluster_data, axis=0)

        distances_to_own_cluster = np.linalg.norm(cluster_data - cluster_mean, axis=1)

        # Calculate the distances to barycenters of other clusters
        distances_to_other_clusters = []
        for j in range(1, K + 1):
            if j != k:
                other_cluster_data = data[labels == j]
                other_cluster_mean = np.mean(other_cluster_data, axis=0)
                distances_to_other_clusters.append(np.linalg.norm(cluster_data - other_cluster_mean, axis=1))

        distances_to_other_clusters = np.vstack(distances_to_other_clusters).T
        min_distances_to_other_clusters = np.min(distances_to_other_clusters, axis=1)

        # Calculate the quotients R(Mi) for each point in the cluster
        quotients = distances_to_own_cluster / min_distances_to_other_clusters

        # Calculate Jk
        Jk = np.maximum(0, 1 - np.mean(quotients))

        wemmert_gancarski_values.append(Jk)

    # Calculate the Wemmert-Gancarski Index
    wemmert_gancarski_index = np.mean(wemmert_gancarski_values)

    return wemmert_gancarski_index
    
def calculate_xie_beni_index(data, labels):
    """
    Calculate the Xie-Beni Index for clustering.

    Parameters:
    - data: A 2D numpy array where each column represents a dimension.
    - labels: A 1D numpy array or list containing the cluster assignments for each data point.

    Returns:
    - xie_beni_index: The Xie-Beni Index value.
    """
    N = len(data)
    K = len(np.unique(labels))
    
    # Number of features
    num_features = data.shape[1]

    # Calculate the within-group sum of squares (WGSS)
    wgss = 0.0

    for k in range(1, K + 1):
        cluster_data = data[labels == k]
        
        if len(cluster_data) > 0:
            
            print("cluster_data: ")
            print(cluster_data)     
            
            cluster_mean = np.mean(cluster_data, axis=0)
            
            wgss = 0       
            
            for point in cluster_data.itertuples(index=False):
                print("point:")
                print(point)
                
                print("cluster_mean: ")
                print(cluster_mean)
                
                x = np.array(point, dtype=np.float64) - np.array(cluster_mean, dtype=np.float64)
                wgss += np.linalg.norm(x)**2
 
    # Calculate the minimum of the minimal squared distances between the points in the clusters
    min_squared_distances = np.inf
    
    print("data: ")
    print(data)
    
    data_list = data.values.tolist()
    data = data_list
    
    print(data[5][5])
 
    for i in range(N):
        for j in range(i+1, N):
            for jx in range(num_features-1):
                if labels[i] == labels[j]:
                    squared_distance = np.linalg.norm(data[i][jx] - data[j][jx])**2
                    min_squared_distances = min(min_squared_distances, squared_distance)

    # Calculate the Xie-Beni Index
    xie_beni_index = wgss / (N * min_squared_distances)
 
    return xie_beni_index 
 
# Function to calculate clustering indices
def calculate_indices(data, labels):
    pairwise_distances = metrics.pairwise_distances(data)
    
    davies_bouldin = metrics.davies_bouldin_score(data, labels)
      
    indices = {
        'Ball-Hall': calculate_ball_hall_manual(data, labels),
        'Banfeld-Raftery': calculate_banfeld_raftery_manual(data, labels),
        'Cindex': calculate_cindex_manual(data, labels),  ## Concordance Index
        'Calinski-Harabasz': calculate_calinski_harabasz_manual(data, labels),
        'Davies-Bouldin': davies_bouldin,
        'Dunn': calculate_dunn_index_manual(data, labels), 
        'Baker-HubertGamma': calculate_baker_hubert_gamma(data, labels),
        'GDI': calculate_gdi(data, labels), # pairwise_distances.mean(metric='mahalanobis'),
        'G_plus':  calculate_g_plus(data, labels),
        'KsqDetW': calculate_ksqdetw(data, labels),  
        'Log_Det_Ratio': calculate_log_det_ratio(data, labels),
        'Log_SS_Ratio': calculate_log_trace_ratio(data, labels),
        'McClain-Rao': calculate_mcclain_rao(data, labels, metric='braycurtis'),
        'PBM': calculate_pbm_index(data, labels),
        'Point-Biserial': calculate_point_bisserial(data, labels),
        'Ratkowsky-Lance': ratkowski_lance_index(data, labels),
        'Ray-Turi': ray_turi_index(data, labels),
        'Scott-Symons': scott_symons_index(data, labels),
        'SD': calculate_SD_index(data, labels),
        'SDbw': calculate_sdbw(data, labels, epsilon=1e-6),
        'Silhouette': calculate_silhouette_index(data, labels),
        'Tau': calculate_tau_index(data, labels),
        'TraceW': calculate_trace_w(data, labels),
        'TraceWiB': calculate_trace_wib(data, labels),
        'Wemmert-Gançarski': calculate_wemmert_gancarski_index(data, labels),
        'Xie-Beni': calculate_xie_beni_index(data, labels),
        # Add more indices as needed
    }
    return indices   
  
# Function to perform clustering and return indices
def perform_clustering(data, n_clusters):
    # Replace KMeans with your preferred clustering algorithm
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)
    indices = calculate_indices(data, labels)
    return indices
 
# Example usage
if __name__ == "__main__":
    # Load your dataset
    # Replace 'your_dataset.csv' with the path to your dataset
 #   data = pd.read_csv('your_dataset.csv')
    
    layout = [
        [sg.Text('Select a CSV file')],
        [sg.InputText(key='file_path', enable_events=True), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))],
        [sg.Button('Submit'), sg.Button('Exit')]
    ]

    window = sg.Window('CSV File Selector', layout, resizable=True)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Submit':
            file_path = values['file_path']
            if file_path.lower().endswith('.csv'):
                try:
                    data = pd.read_csv(file_path)
                    sg.popup('CSV File Loaded Successfully!', title='Success')
                    break
                except Exception as e:
                    sg.popup_error(f'Error loading CSV file: {str(e)}', title='Error')
            else:
                sg.popup_error('Please select a valid CSV file.', title='Error')

    window.close()
    

    # Specify the number of clusters
    n_clusters = 3  # Adjust as needed

    # Perform clustering and get indices
    clustering_indices = perform_clustering(data, n_clusters)

    keys_list = list(clustering_indices.keys())    
    removed_key = keys_list[3]
    del clustering_indices[removed_key]

    # Display and save results to Excel  
    
    print("clustering_indices: ")
    print(clustering_indices) 
    
 ##   import math
##    cleaned_dict = {}
    
    # print("clustering_indices.items()")
    # print(clustering_indices.items())
    # Create a DataFrame from the dictionary
    # Create a DataFrame from the dictionary
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(list(clustering_indices.items()), columns=['Metric', 'Value'])
        
    # Replace NaN values with a placeholder (e.g., 0) to avoid RuntimeWarning
    df['Value'].fillna(0, inplace=True)
    
    # Drop rows with values equal to 0
    df = df[df['Value'] != 0]
    
    # Normalize the remaining values
    scaler = MinMaxScaler() 
    df['Normalized_Value'] = scaler.fit_transform(df[['Value']])
    
    # Drop rows with values equal to 0
    df = df[df['Normalized_Value'] != 0]
    
    # Drop the original 'Value' column if you only want to keep the normalized values
    df.drop(columns=['Value'], inplace=True)   
    
    # Specify the file name
    excel_file_name = "clustering_post_analysis.xlsx"
    
    import os
    
    if os.path.exists(excel_file_name):
        os.remove(excel_file_name)
        print(f"Existing file {excel_file_name} has been deleted.")      
    
    # Write the DataFrame to an Excel file
    df.to_excel(excel_file_name, index=False)
    
    print(f"DataFrame has been written to {excel_file_name}")