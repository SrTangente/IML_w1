from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from evaluate import evaluate_DBSCAN

def compareDBSCAN_alg(data, classes, eps):
    data = StandardScaler().fit_transform(data)
    auto_clustering = DBSCAN(eps=eps, algorithm='auto').fit(data)
    print('Auto:')
    print(auto_clustering.labels_)
    evaluate_DBSCAN(auto_clustering, data, classes)
    ball_tree_clustering = DBSCAN(algorithm='ball_tree', eps=eps).fit(data)
    print('Ball tree:')
    print(ball_tree_clustering.labels_)
    evaluate_DBSCAN(ball_tree_clustering, data, classes)
    kd_tree_clustering = DBSCAN(algorithm='kd_tree', eps=eps).fit(data)
    print('KD tree:')
    print(kd_tree_clustering.labels_)
    evaluate_DBSCAN(kd_tree_clustering, data, classes)
    try:
        brute_clustering = DBSCAN(algorithm='brute', eps=eps, metric='cosine').fit(data)
        print('Brute:')
        print(brute_clustering.labels_)
        evaluate_DBSCAN(brute_clustering, data, classes)
    except:
        print("Brute clustering could not be performed (too may data)")

def compareDBSCAN_metric(data, classes, eps):
    data = StandardScaler().fit_transform(data)
    euclidean_clustering = DBSCAN(metric='euclidean', eps=eps).fit(data)
    print('Euclidean:')
    print(euclidean_clustering.labels_)
    evaluate_DBSCAN(euclidean_clustering, data, classes)
    cosine_clustering = DBSCAN(metric='cosine', eps=eps).fit(data)
    print('Cosine:')
    print(cosine_clustering.labels_)
    evaluate_DBSCAN(cosine_clustering, data, classes)
    l1_clustering = DBSCAN(metric='l1', eps=eps).fit(data)
    print('L1:')
    print(l1_clustering.labels_)
    evaluate_DBSCAN(cosine_clustering, data,classes)
    l2_clustering = DBSCAN(metric='l2', eps=eps).fit(data)
    print('L2:')
    print(l2_clustering.labels_)
    evaluate_DBSCAN(cosine_clustering, data, classes)
    man_clustering = DBSCAN(metric='manhattan', eps=eps).fit(data)
    print('Manhattan:')
    print(man_clustering.labels_)
    evaluate_DBSCAN(cosine_clustering, data, classes)

