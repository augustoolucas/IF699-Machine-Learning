from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.io.arff import loadarff

import pandas as pd
import numpy as np
import time
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import knn

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
    attributes, labels = load_dataset('./kc2.arff')
    label_column = get_data_label(labels)

    attributes_bugs = attributes[labels[label_column] == 1]
    labels_bugs = labels[labels[label_column] == 1]

    attributes_non_bugs = attributes[labels[label_column] == 0]
    labels_non_bugs = labels[labels[label_column] == 0]

    for split_ratio in [0.3, 0.4, 0.5]:
        precision = []
        recall = []
        f1 = []
        for threshold in range(0*len(attributes.columns)//2, len(attributes.columns)+1):
            x_train = attributes_non_bugs.iloc[0:int(len(attributes_non_bugs)*split_ratio)].reset_index(drop=True)

            x_test = attributes_non_bugs.iloc[int(len(attributes_non_bugs)*split_ratio):].reset_index(drop=True)
            x_test = x_test.append(attributes_bugs).reset_index(drop=True)

            y_test = labels_non_bugs.iloc[int(len(labels_non_bugs)*split_ratio):].reset_index(drop=True)
            y_test = y_test.append(labels_bugs).reset_index(drop=True)

            x_train_avg = x_train.mean().to_frame().transpose()
            x_train_stdev = x_train.std().to_frame().transpose()
            x_test_delta = x_test.sub(x_train_avg.iloc[0, :]).abs()
            less_than_stdev = x_test_delta.le(x_train_stdev.iloc[0, :])
            prediction = (less_than_stdev.sum(1) < (threshold +1)).astype(int)

            precision.append(metrics.precision_score(y_test, prediction))
            recall.append(metrics.recall_score(y_test, prediction))
            f1.append(metrics.f1_score(y_test, prediction))

        print('Split:', split_ratio)
        print('Precision:', precision)
        print('F1', f1)
        print()
        plt.figure()
        plt.plot(range(0*len(attributes.columns)//2 + 1, len(attributes.columns)+2), precision, label='Precision')
        #plt.plot(range(0*len(attributes.columns)//2, len(attributes.columns)+1), recall, label='Recall')
        plt.plot(range(0*len(attributes.columns)//2 + 1, len(attributes.columns)+2), f1, label='F1')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(fname=f'{split_ratio}_split_results.png',
                    dpi=144)
