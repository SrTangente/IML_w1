from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from evaluate import evaluate_DBSCAN

def compareDBSCAN_alg(data, classes, eps, minPts):
    data = StandardScaler().fit_transform(data)
    auto_clustering = DBSCAN(eps=eps, algorithm='auto', min_samples=minPts).fit(data)
    print('Auto:')
    evaluate_DBSCAN(auto_clustering.labels_, data, classes)
    ball_tree_clustering = DBSCAN(algorithm='ball_tree', eps=eps, min_samples=minPts).fit(data)
    print('Ball tree:')
    evaluate_DBSCAN(ball_tree_clustering.labels_, data, classes)
    kd_tree_clustering = DBSCAN(algorithm='kd_tree', eps=eps, min_samples=minPts).fit(data)
    print('KD tree:')
    evaluate_DBSCAN(kd_tree_clustering.labels_, data, classes)
    try:
        brute_clustering = DBSCAN(algorithm='brute', eps=eps, min_samples=minPts).fit(data)
        print('Brute:')
        evaluate_DBSCAN(brute_clustering.labels_, data, classes)
    except:
        print("Brute clustering could not be performed (too may data)")

def compareDBSCAN_metric(data, classes, eps, minPts):
    data = StandardScaler().fit_transform(data)
    euclidean_clustering = DBSCAN(metric='euclidean', eps=eps, min_samples=minPts).fit(data)
    print('Euclidean:')
    evaluate_DBSCAN(euclidean_clustering.labels_, data, classes)
    try:
        cosine_clustering = DBSCAN(metric='cosine', eps=eps, min_samples=minPts).fit(data)
        print('Cosine:')
        evaluate_DBSCAN(cosine_clustering.labels_, data, classes)
    except:
        print("Cosine clustering could not be performed (too may data)")
    man_clustering = DBSCAN(metric='manhattan', eps=eps, min_samples=minPts).fit(data)
    print('Manhattan:')
    evaluate_DBSCAN(man_clustering.labels_, data, classes)
    return

