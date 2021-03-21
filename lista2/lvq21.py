from sklearn.neighbors import KNeighborsClassifier

import lvq_common
import numpy as np

e = 1e-12

def gen_prototypes(x, y):
    protos_x, protos_y = lvq_common.get_random_prototypes(x, y, 20)
    classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(protos_x, protos_y)

    neighbors_proto = classifier.kneighbors(X=x, n_neighbors=2,
                                            return_distance=False)

    for _ in range(lvq_common.NUM_UPDATES):
        for idx in range(len(x)):
            instance = x.iloc[idx]
            instance_class = y.iloc[idx].values[0]

            nbs = neighbors_proto[idx]
            p1_idx, p2_idx = nbs[0], nbs[1]

            p1_x = protos_x.iloc[p1_idx]
            p1_y = protos_y[p1_idx]
            p1_distance = np.linalg.norm(p1_x - instance)

            p2_x = protos_x.iloc[p2_idx]
            p2_y = protos_y[p2_idx]
            p2_distance = np.linalg.norm(p2_x - instance)

            distance12 = p1_distance / (p2_distance + e)
            distance21 = p2_distance / (p1_distance + e)

            if min(distance12, distance21) > lvq_common.WINDOW:
                if p1_y != p2_y:
                    weight = lvq_common.WEIGHT if p1_y == instance_class else -lvq_common.WEIGHT
                    lvq_common.update_prototype(p1_x, instance, weight)

                    weight = lvq_common.WEIGHT if p2_y == instance_class else -lvq_common.WEIGHT
                    lvq_common.update_prototype(p2_x, instance, weight)

    return protos_x, protos_y
