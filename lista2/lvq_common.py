import random

NUM_UPDATES = 25
WEIGHT = 0.1
WINDOW = (1 - WEIGHT) / (1 + WEIGHT)

def get_random_prototypes(x, y, num_gen):
    idxs = []
    labels = {x[0] for x in y.values.tolist()}

    for label in labels:
        for _ in range(int(num_gen/len(labels))):
            while True:
                idx = random.randint(0, len(x) - 1)

                # if (y.iloc[idx] != label).values[0]:
                #    continue
                if idx in idxs:
                    continue
                else:
                    idxs.append(idx)
                    break

    random.shuffle(idxs)
    random_x = x.iloc[idxs]
    random_y = y.iloc[idxs]
    random_x = random_x.reset_index(drop=True)
    random_y = random_y.reset_index(drop=True)
    random_y = [x[0] for x in random_y.values.tolist()]

    return random_x, random_y


def update_prototype(proto_x, instance, w):
    for idx, (proto_attr, instance_attr) in enumerate(zip(proto_x, instance.iteritems())):
        instance_attr = instance_attr[1]  # df.iteritems() returns (label, content)
        weighted_diff = w * (instance_attr - proto_attr)
        proto_x[idx] = proto_attr + weighted_diff

    return proto_x
