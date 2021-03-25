from sklearn.neighbors import KNeighborsClassifier

import lvq_common as lvqc
import numpy as np

e = 1e-12

def gen_prototypes(x, y, num_protos):
    protos_x, protos_y = lvqc.get_random_prototypes(x, y, num_protos)
    classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(protos_x, protos_y)
    neighbors_proto = classifier.kneighbors(X=x, n_neighbors=2,
                                            return_distance=False)

    for _ in range(lvqc.NUM_UPDATES):
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

            if min(distance12, distance21) > lvqc.WINDOW:
                if p1_y != p2_y:
                    weight = lvqc.WEIGHT if p1_y == instance_class else -lvqc.WEIGHT
                    lvqc.update_prototype(p1_x, instance, weight)

                    weight = lvqc.WEIGHT if p2_y == instance_class else -lvqc.WEIGHT
                    lvqc.update_prototype(p2_x, instance, weight)
                else:
                    lvqc.update_prototype(p1_x, instance, lvqc.WEIGHT)
                    lvqc.update_prototype(p2_x, instance, lvqc.WEIGHT)
                    
    return protos_x, protos_y
