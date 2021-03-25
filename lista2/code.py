import knn
import time
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import lvq1
import lvq21
import lvq31

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.io.arff import loadarff


def get_data_label(df):
    return df.columns.values[-1]
    

def load_dataset(filename):
    arff_data = loadarff(filename)
    df = pd.DataFrame(arff_data[0])
    df = df.dropna()
    df = df.sample(frac=1).reset_index(drop=True)

    label_column = get_data_label(df)
    attributes = pd.DataFrame(df.drop(label_column, 1))
    labels = pd.DataFrame(df[label_column])

    return attributes, labels


class Metrics():
    def __init__(self):
        self.accuracy = {}
        self.precision = {}
        self.recall = {}
        self.f1 = {}
        self.test_time = {}
        self.training_time = {}

    def update(self, k, y_test, y_pred):
        self.update_acc(k, metrics.accuracy_score(y_test, y_pred))
        self.update_precision(k, metrics.precision_score(y_test, y_pred))
        self.update_recall(k, metrics.recall_score(y_test, y_pred))
        self.update_f1(k, metrics.f1_score(y_test, y_pred))

    def update_acc(self, k, value):
        if k not in self.accuracy:
            self.accuracy[k] = [value]
        else:
            self.accuracy[k].append(value)

    def update_precision(self, k, value):
        if k not in self.precision:
            self.precision[k] = [value]
        else:
            self.precision[k].append(value)

    def update_recall(self, k, value):
        if k not in self.recall:
            self.recall[k] = [value]
        else:
            self.recall[k].append(value)

    def update_f1(self, k, value):
        if k not in self.f1:
            self.f1[k] = [value]
        else:
            self.f1[k].append(value)

    def update_test_time(self, k, value):
        if k not in self.test_time:
            self.test_time[k] = [value]
        else:
            self.test_time[k].append(value)

    def update_training_time(self, k, value):
        if k not in self.training_time:
            self.training_time[k] = [value]
        else:
            self.training_time[k].append(value)

def plot_data(data, filename, xlabel='', ylabel=''):
    fig = plt.figure()
    for k, v in data.items():
        plt.plot(list(range(1, 6)), v, label=f'K = {k}')

    lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xticks(range(1, 6))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(filename,
            dpi=144,
            format='png',
            bbox_extra_artists=(lg,),
            bbox_inches='tight')

def plot_data_avg(data, metric, filename, xlabel='', ylabel='', label=''):
    fig = plt.figure()
    min_v = 999
    max_v = 0
    for k, proto_metrics in prototypes_metrics.items():
        v = []
        if metric == 'acc':
            data = proto_metrics[0].accuracy
        elif metric == 'prec':
            data = proto_metrics[0].precision
        elif metric == 'f1':
            data = proto_metrics[0].f1
        elif metric == 'train_time':
            data = proto_metrics[0].training_time
        elif metric == 'test_time':
            data = proto_metrics[0].test_time

        v.append(list(map(avg, list(data.values()))))
        v = v[0]
        p, = plt.plot(range(1, len(v)+1), v, alpha=0.8, linewidth=3, label=f'K = {k}')

        #if len(data) > 1:
            #p.set_label(k.capitalize())

        plt.xticks(range(1, len(data.keys())+1), data.keys())

        if min_v > min(v):
            min_v = min(v)

        if max_v < max(v):
            max_v = max(v)

    plt.xlabel(xlabel, fontsize='x-large')
    plt.ylabel(ylabel, fontsize='x-large')
    plt.ylim(min_v*0.95, max_v*1.05)

    if len(data) > 1:
        lg = plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=len(data),
                        borderaxespad=0, frameon=False, fontsize='x-large')
        plt.savefig(filename,
                    dpi=144,
                    format='png',
                    bbox_extra_artists=(lg,),
                    bbox_inches='tight')
    else:
        plt.legend()
        plt.savefig(filename,
                    dpi=144,
                    format='png')


def avg(lst):
    return sum(lst)/len(lst)


if __name__ == '__main__':
    attributes, labels = load_dataset('./kc1.arff')
    label_column = get_data_label(labels)

    if label_column == 'defects':
        labels = labels.astype(str).replace({"b'false'": 0, "b'true'": 1})
    else:
        labels = labels.astype(str).replace({"b'no'": 0, "b'yes'": 1})

    kf = KFold(n_splits=5)
    kf.get_n_splits(attributes)

    k_values = range(1, 4)
    num_prototypes = [100, 200, 300]
    prototypes_metrics = {}

    for n_k in k_values:
        prototypes_metrics[n_k] = []
        knn_uniform_metrics = Metrics()
        print('K Value: ', n_k)
        for num_protos in num_prototypes:
            print('Num Protos:', num_protos)
            for train_idx, test_idx in kf.split(attributes):
                x_train = attributes.iloc[train_idx].reset_index(drop=True)
                x_test = attributes.iloc[test_idx].reset_index(drop=True)
                y_train = labels.iloc[train_idx].reset_index(drop=True)
                y_test = labels.iloc[test_idx].reset_index(drop=True)

                x_train, y_train = lvq1.gen_prototypes(x_train,
                                                       y_train,
                                                       num_protos)
                y_data = {label_column: y_train}
                y_train = pd.DataFrame.from_dict({label_column: y_train})

                x_train, y_train = lvq21.gen_prototypes(x_train,
                                                        y_train,
                                                            num_protos)

                y_data = {label_column: y_train}
                y_train = pd.DataFrame.from_dict({label_column: y_train})

                scaler = StandardScaler().fit(x_train)
                x_train = scaler.transform(x_train)

                scaler = StandardScaler().fit(x_test)
                x_test = scaler.transform(x_test)

                y_train = y_train[label_column].tolist()
                y_test = y_test[label_column].tolist()

                start_time = time.time()
                classifier = KNeighborsClassifier(n_neighbors=n_k,
                                                  weights='uniform')

                classifier.fit(x_train, y_train)
                knn_uniform_metrics.update_training_time(num_protos,
                                                         time.time() - start_time)

                start_time = time.time()
                y_pred = classifier.predict(x_test)
                knn_uniform_metrics.update_test_time(num_protos,
                                                     time.time() - start_time)

                knn_uniform_metrics.update(num_protos, y_test, y_pred)
        prototypes_metrics[n_k].append(knn_uniform_metrics)


    plot_data_avg(prototypes_metrics, 'acc', f'lvq2_accuracy_avg.png', 'Num of Protos', 'Avg Accuracy')
    plot_data_avg(prototypes_metrics, 'prec', f'lvq2_precision_avg.png', 'Num of Protos', 'Avg Accuracy')
    plot_data_avg(prototypes_metrics, 'f1', f'lvq2_f1_avg.png', 'Num of Protos', 'Avg Accuracy')
    plot_data_avg(prototypes_metrics, 'train_time', f'lvq2_f1_avg.png', 'Num of Protos', 'Avg Accuracy')
    plot_data_avg(prototypes_metrics, 'test_time', f'lvq2_time_avg.png', 'Num of Protos', 'Avg Accuracy')
"""
print('Uniform Avg Training Time: ',
      avg(list(map(avg, list(knn_uniform_metrics.training_time.values())))))
print('Uniform Avg Test Time: ',
      avg(list(map(avg, list(knn_uniform_metrics.test_time.values())))))
"""
