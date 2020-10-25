import numpy as np


def kmeans(data, k):
    n = np.size(data, 0)
    n_prop = np.size(data, 1)
    tagged_data = np.ones([n, n_prop + 1])
    tagged_data = -1 * tagged_data
    tagged_data[:, :-1] = data
    # we add a column for the classification
    seeds = []
    for l in range(k):
        r = np.random.randint(0, n)
        new_seed = data[r, :]
        ids = map(id, seeds)
        if id(new_seed) not in ids:
            seeds.append(new_seed)
        tagged_data[r, -1] = l
    return kmeans_core(tagged_data, seeds)


def kmeans_core(tagged_data, seeds):
    n = np.size(tagged_data, 0)
    n_prop = np.size(tagged_data, 1) - 1
    # save tags state
    tags = np.copy(tagged_data[:, -1])
    # start loop while
    while True:
        for i in range(n):
            min_distance = None
            for j in range(len(seeds)):
                # calculate distance for each centroid
                distance = np.linalg.norm(seeds[j] - tagged_data[i, :-1], ord=2)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    tag = j
                    # save closest centroid index in the tag column
                    tagged_data[i, -1] = tag
        new_tags = tagged_data[:, -1]
        if (new_tags == tags).all():
            # if tags have not changed since last iteration, we are done
            return tagged_data
        else:
            # if not, centroids are recalculated and tags updated
            tags = np.copy(new_tags)
            for s in range(len(seeds)):
                close_points = [x[:-1] for x in tagged_data if x[-1] == s]
                centroid = calculate_centroid(close_points)
                seeds[s] = centroid


def calculate_centroid(close_points):
    centroid = np.mean(close_points, axis=0)
    return centroid
