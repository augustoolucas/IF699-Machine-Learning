from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from scipy.io.arff import loadarff

import pandas as pd
import numpy as np
import time
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def get_data_label(df):
    return df.columns.values[-1]


def load_dataset(filename):
    arff_data = loadarff(filename)
    df = pd.DataFrame(arff_data[0])
    df = df.dropna()
    df = df.sample(frac=1).reset_index(drop=True)

    label_column = get_data_label(df)

    if label_column == 'defects':
        df[label_column] = df[label_column].astype(str).replace({"b'false'": 0, "b'true'": 1})
    else:
        df[label_column] = df[label_column].astype(str).replace({"b'no'": 0, "b'yes'": 1})

    # df = df[(df[label_column] == 0)].reset_index(drop=True)

    features = pd.DataFrame(df.drop(label_column, 1))
    labels = pd.DataFrame(df[label_column])

    return features, labels


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


def get_best_k(data, k_values):
    silhouette = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k).fit(data)
        silhouette_score = metrics.silhouette_score(data, kmeans.labels_)
        silhouette.append(silhouette_score)
    
    best_k = k_values[silhouette.index(max(silhouette))]
    return best_k


def get_train_data(data1, data2, features):
    x_data = []
    y_data = []

    for data, label in [data1, data2]:
        for centroid in data.cluster_centers_:
            aux_x = {c: centroid[idx] for idx, c in enumerate(features)}

            x_data.append(aux_x)
            y_data.append(label)

    x_data = pd.DataFrame.from_dict(x_data)
    y_data = np.asarray(y_data)

    return x_data, y_data


if __name__ == '__main__':
    features, labels = load_dataset('./kc1.arff')
    label_column = get_data_label(labels)

    features_bugs = features[labels[label_column] == 1]
    labels_bugs = labels[labels[label_column] == 1]

    features_no_bugs = features[labels[label_column] == 0]
    labels_no_bugs = labels[labels[label_column] == 0]

    k_values = list(range(2, 7, 1))

    x_train_bugs = features_bugs.iloc[0:int(len(features_bugs)*0.8)].reset_index(drop=True)
    y_train_bugs = labels_bugs.iloc[0:int(len(labels_bugs)*0.8)].reset_index(drop=True)

    x_train_no_bugs = features_no_bugs.iloc[0:int(len(features_no_bugs)*0.8)].reset_index(drop=True)
    y_train_no_bugs = labels_no_bugs.iloc[0:int(len(labels_no_bugs)*0.8)].reset_index(drop=True)

    x_test = features_no_bugs.iloc[int(len(features_no_bugs)*0.8):].reset_index(drop=True)
    x_test = x_test.append(features_bugs.iloc[int(len(features_bugs)*0.8):]).reset_index(drop=True)

    y_test = labels_no_bugs.iloc[int(len(labels_no_bugs)*0.8):].reset_index(drop=True)
    y_test = y_test.append(labels_bugs.iloc[int(len(labels_bugs)*0.8):]).reset_index(drop=True)

    k_bugs = get_best_k(x_train_bugs, k_values)
    k_no_bugs = get_best_k(x_train_no_bugs, k_values)

    kmeans_bugs = KMeans(n_clusters=k_bugs).fit(x_train_bugs)
    kmeans_no_bugs = KMeans(n_clusters=k_no_bugs).fit(x_train_no_bugs)

    x_train, y_train = get_train_data((kmeans_bugs, True),
                                      (kmeans_no_bugs, False),
                                      features.columns)

    naive_bayes = GaussianNB().fit(x_train, y_train)
    prediction = naive_bayes.predict(x_test)
    print(metrics.classification_report(y_test, prediction))
