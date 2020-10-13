from sklearn.cluster import DBSCAN

def compareDBSCAN(data):
    ball_tree_clustering = DBSCAN(algorithm='ball_tree').fit(data)
    print('Ball tree:')
    print(ball_tree_clustering.labels_)
    kd_tree_clustering = DBSCAN(algorithm='kd_tree').fit(data)
    print('KD tree:')
    print(kd_tree_clustering.labels_)
    try:
        brute_clustering = DBSCAN(algorithm='brute').fit(data)
        print('Brute:')
        print(brute_clustering.labels_)
    except:
        print("Brute clustering could not be performed (too many data)")