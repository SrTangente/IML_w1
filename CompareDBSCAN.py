from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from evaluate import evaluate_clustering

def compareDBSCAN_alg(data, eps):
    data = StandardScaler().fit_transform(data)
    auto_clustering = DBSCAN(eps=eps, algorithm='auto').fit(data)
    print('Auto:')
    print(auto_clustering.labels_)
    evaluate_clustering(auto_clustering, data)
    ball_tree_clustering = DBSCAN(algorithm='ball_tree', eps=eps).fit(data)
    print('Ball tree:')
    print(ball_tree_clustering.labels_)
    evaluate_clustering(ball_tree_clustering, data)
    kd_tree_clustering = DBSCAN(algorithm='kd_tree', eps=eps).fit(data)
    print('KD tree:')
    print(kd_tree_clustering.labels_)
    evaluate_clustering(kd_tree_clustering, data)
    try:
        brute_clustering = DBSCAN(algorithm='brute', eps=eps, metric='cosine').fit(data)
        print('Brute:')
        print(brute_clustering.labels_)
        evaluate_clustering(brute_clustering, data)
    except:
        print("Brute clustering could not be performed (too may data)")

def compareDBSCAN_metric(data, eps):
    data = StandardScaler().fit_transform(data)
    euclidean_clustering = DBSCAN(metric='euclidean', eps=eps).fit(data)
    print('Euclidean:')
    print(euclidean_clustering.labels_)
    evaluate_clustering(euclidean_clustering, data)
    cosine_clustering = DBSCAN(metric='cosine', eps=eps).fit(data)
    print('Cosine:')
    print(cosine_clustering.labels_)
    evaluate_clustering(cosine_clustering, data)
    l1_clustering = DBSCAN(metric='l1', eps=eps).fit(data)
    print('L1:')
    print(l1_clustering.labels_)
    evaluate_clustering(cosine_clustering, data)
    l2_clustering = DBSCAN(metric='l2', eps=eps).fit(data)
    print('L2:')
    print(l2_clustering.labels_)
    evaluate_clustering(cosine_clustering, data)
    man_clustering = DBSCAN(metric='manhattan', eps=eps).fit(data)
    print('Manhattan:')
    print(man_clustering.labels_)
    evaluate_clustering(cosine_clustering, data)
