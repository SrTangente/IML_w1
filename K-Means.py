import numpy as np

def kmeans(data, k):
    n = np.size(data, 0)
    n_prop = np.size(data, 1)
    tagged_data = np.zeros([n, n_prop+1])
    tagged_data[:, 0:-1] = data
    seeds = []
    for l in range(k):
        r = np.random.randint(0,n)
        seeds.append(data[r, :])
        tagged_data[r, -1] = l

    for i in range(n):
        min_distance = None
        closer_seed = None
        tag = None
        for j in range(len(seeds)):
            distance = np.linalg.norm(seeds[j] - data[i,:], ord=n_prop)
            if distance is None or distance < min_distance:
                min_distance = distance
                tag = j
                tagged_data[i, -1] = tag

    #repeat the process several iterations until tags column is not changed anymore