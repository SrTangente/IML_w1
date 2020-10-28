from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from evaluate import evaluate_DBSCAN

def compareDBSCAN_alg(data, classes, eps, minPts):
    data = StandardScaler().fit_transform(data)
    auto_clustering = DBSCAN(eps=eps, algorithm='auto', min_samples=minPts).fit(data)
    print('Auto:')
    print(auto_clustering.labels_)
    evaluate_DBSCAN(auto_clustering.labels_, data, classes)
    ball_tree_clustering = DBSCAN(algorithm='ball_tree', eps=eps, min_samples=minPts).fit(data)
    print('Ball tree:')
    print(ball_tree_clustering.labels_)
    evaluate_DBSCAN(ball_tree_clustering.labels_, data, classes)
    kd_tree_clustering = DBSCAN(algorithm='kd_tree', eps=eps, min_samples=minPts).fit(data)
    print('KD tree:')
    print(kd_tree_clustering.labels_)
    evaluate_DBSCAN(kd_tree_clustering.labels_, data, classes)
    try:
        brute_clustering = DBSCAN(algorithm='brute', eps=eps, min_samples=minPts).fit(data)
        print('Brute:')
        print(brute_clustering.labels_)
        evaluate_DBSCAN(brute_clustering.labels_, data, classes)
    except:
        print("Brute clustering could not be performed (too may data)")

def compareDBSCAN_metric(data, classes, eps, minPts):
    data = StandardScaler().fit_transform(data)
    clusters_num = np.ones([5])*-1
    euclidean_clustering = DBSCAN(metric='euclidean', eps=eps, min_samples=minPts).fit(data)
    print('Euclidean:')
    print(euclidean_clustering.labels_)
    evaluate_DBSCAN(euclidean_clustering.labels_, data, classes)
    try:
        cosine_clustering = DBSCAN(metric='cosine', eps=eps, min_samples=minPts).fit(data)
        print('Cosine:')
        print(cosine_clustering.labels_)
        evaluate_DBSCAN(cosine_clustering.labels_, data, classes)
    except:
        print("Cosine clustering could not be performed (too may data)")
    l1_clustering = DBSCAN(metric='l1', eps=eps, min_samples=minPts).fit(data)
    print('L1:')
    print(l1_clustering.labels_)
    evaluate_DBSCAN(l1_clustering.labels_, data, classes)
    l2_clustering = DBSCAN(metric='l2', eps=eps, min_samples=minPts).fit(data)
    print('L2:')
    print(l2_clustering.labels_)
    evaluate_DBSCAN(l2_clustering.labels_, data, classes)
    man_clustering = DBSCAN(metric='manhattan', eps=eps, min_samples=minPts).fit(data)
    print('Manhattan:')
    print(man_clustering.labels_)
    evaluate_DBSCAN(man_clustering.labels_, data, classes)
    return

