from sklearn.neighbors import KNeighborsClassifier

import lvq_common

def gen_prototypes(x, y, num_protos):
    protos_x, protos_y = lvq_common.get_random_prototypes(x, y, num_protos)
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(x, y)

    predictions = classifier.predict(protos_x)
    neighbors_proto = classifier.kneighbors(X=protos_x,
                                            n_neighbors=1,
                                            return_distance=False)

    for _ in range(lvq_common.NUM_UPDATES):
        for (proto_x, proto_y, pred, nb) in zip(protos_x.iterrows(), protos_y, predictions, neighbors_proto):
            proto_x = proto_x[1]  # df.iterrows() returns (index, row)
            instance = x.iloc[nb]
            breakpoint()
            weight = lvq_common.WEIGHT if proto_y == pred else -lvq_common.WEIGHT
            proto_x = lvq_common.update_prototype(proto_x, instance, weight)

    return protos_x, proto_y
