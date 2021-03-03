import time
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.io.arff import loadarff
import knn

def load_dataset(filename):
    arff_data = loadarff(filename)
    df = pd.DataFrame(arff_data[0])
    df = df.dropna()
    df = df.sample(frac=1).reset_index(drop=True)

    attributes = pd.DataFrame(df.drop('defects', 1))
    labels = pd.DataFrame(df['defects'])

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

def plot_data_avg(data, filename, xlabel='', ylabel=''):
    fig = plt.figure()
    min_v = 999
    max_v = 0
    for k in data:
        v = list(map(avg, list(data[k].values())))
        plt.plot(range(1, len(v)+1), v, label=k.capitalize())
        plt.xticks(range(1, len(data[k].keys())+1), data[k].keys())

        if min_v > min(v):
            min_v = min(v)

        if max_v < max(v):
            max_v = max(v)

    lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(min_v*0.95, max_v*1.05)

    plt.savefig(filename,
                dpi=144, 
                format='png',
                bbox_extra_artists=(lg,),
                bbox_inches='tight')


def avg(lst):
    return sum(lst)/len(lst)


if __name__ == '__main__':
    attributes, labels = load_dataset('./kc1.arff')
    labels = labels.astype(str).replace({"b'false'": 0, "b'true'": 1})
    #labels = labels.astype(str).replace({"b'no'": 0, "b'yes'": 1})

    kf = KFold(n_splits=5)
    kf.get_n_splits(attributes)

    knn_uniform_metrics = Metrics()
    knn_weighted_metrics = Metrics()
    knn_adaptive_metrics = Metrics()

    k_values = [1, 2, 3, 5, 7, 9, 11, 13, 15]

    for n_k in k_values:
        for train_idx, test_idx in kf.split(attributes):
            x_train, x_test = attributes.iloc[train_idx], attributes.iloc[test_idx]
            y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

            """
            print('Train')
            print(f'False: {len(y_train) - y_train["defects"].sum()}')
            print(f'True: {y_train["defects"].sum()}')

            print('Test:')
            print(f'False: {len(y_test) - y_test["defects"].sum()}')
            print(f'True: {y_test["defects"].sum()}')
            """
            
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)

            scaler = StandardScaler().fit(x_test)
            x_test = scaler.transform(x_test)

            y_train = y_train['defects'].tolist()
            y_test = y_test['defects'].tolist()

            start_time = time.time()
            classifier = KNeighborsClassifier(n_neighbors=n_k, weights='uniform')
            classifier.fit(x_train, y_train)
            knn_uniform_metrics.update_training_time(n_k, time.time() - start_time)

            start_time = time.time()
            y_pred = classifier.predict(x_test)
            knn_uniform_metrics.update_test_time(n_k, time.time() - start_time)
            knn_uniform_metrics.update(n_k, y_test, y_pred)

            start_time = time.time()
            classifier = KNeighborsClassifier(n_neighbors=n_k, weights='distance')
            classifier.fit(x_train, y_train)
            knn_weighted_metrics.update_training_time(n_k, time.time() - start_time)

            start_time = time.time()
            y_pred = classifier.predict(x_test)
            knn_weighted_metrics.update_test_time(n_k, time.time() - start_time)
            knn_weighted_metrics.update(n_k, y_test, y_pred)

            start = time.time()
            classifier = knn.KNN(n_k, ktype='adaptive')
            classifier.fit(x_train, y_train)
            knn_adaptive_metrics.update_training_time(n_k, time.time() - start_time)

            start = time.time()
            y_pred = classifier.predict(x_test)
            knn_adaptive_metrics.update_test_time(n_k, time.time() - start_time)
            knn_adaptive_metrics.update(n_k, y_test, y_pred)

    data = {}
    data['uniform'] = knn_uniform_metrics.accuracy
    data['weighted'] = knn_weighted_metrics.accuracy
    data['adaptive'] = knn_adaptive_metrics.accuracy
    """
    plot_data(data['uniform'], 'knn_accuracy_uniform.png', 'Fold', 'Accuracy')
    plot_data(data['weighted'], 'knn_accuracy_weighted.png', 'Fold', 'Accuracy')
    plot_data(data['adaptive'], 'knn_accuracy_adaptive.png', 'Fold', 'Accuracy')
    """
    plot_data_avg(data, 'knn_all_accuracy_avg.png', 'Value of K', 'Accuracy')

    data['uniform'] = knn_uniform_metrics.precision
    data['weighted'] = knn_weighted_metrics.precision
    data['adaptive'] = knn_adaptive_metrics.precision
    """
    plot_data(data['uniform'], 'knn_precision_uniform.png', 'Fold', 'Precision')
    plot_data(data['weighted'], 'knn_precision_weighted.png', 'Fold', 'Precision')
    plot_data(data['adaptive'], 'knn_precision_adaptive.png', 'Fold', 'Precision')
    """
    plot_data_avg(data, 'knn_all_precision_avg.png', 'Value of K', 'Precision')

    data['uniform'] = knn_uniform_metrics.recall
    data['weighted'] = knn_weighted_metrics.recall
    data['adaptive'] = knn_adaptive_metrics.recall
    plot_data_avg(data, 'knn_all_recall_avg.png', 'Value of K', 'Recall')

    data['uniform'] = knn_uniform_metrics.f1
    data['weighted'] = knn_weighted_metrics.f1
    data['adaptive'] = knn_adaptive_metrics.f1
    plot_data_avg(data, 'knn_all_f1_avg.png', 'Value of K', 'F1')

    data = {}
    data['uniform'] = knn_uniform_metrics.training_time
    plot_data_avg(data, 'knn_uniform_training_avg.png', 'Value of K', 'Time')
    data = {}
    data['weighted'] = knn_weighted_metrics.training_time
    plot_data_avg(data, 'knn_weighted_training_avg.png', 'Value of K', 'Time')
    data = {}
    data['adaptive'] = knn_adaptive_metrics.training_time
    plot_data_avg(data, 'knn_adaptive_training_avg.png', 'Value of K', 'Time')

    data = {}
    data['uniform'] = knn_uniform_metrics.test_time
    plot_data_avg(data, 'knn_uniform_test_avg.png', 'Value of K', 'Time')
    data = {}
    data['weighted'] = knn_weighted_metrics.test_time
    plot_data_avg(data, 'knn_weighted_test_avg.png', 'Value of K', 'Time')
    data = {}
    data['adaptive'] = knn_adaptive_metrics.test_time
    plot_data_avg(data, 'knn_adaptive_test_avg.png', 'Value of K', 'Time')

    print('Uniform Avg Training Time: ',
          avg(list(map(avg, list(knn_uniform_metrics.training_time.values())))))
    print('Uniform Avg Test Time: ',
          avg(list(map(avg, list(knn_uniform_metrics.test_time.values())))))
    print('Weighted Avg Training Time: ',
          avg(list(map(avg, list(knn_weighted_metrics.training_time.values())))))
    print('Weighted Avg Test Time: ',
          avg(list(map(avg, list(knn_weighted_metrics.test_time.values())))))
    print('Adaptive Avg Training: ',
          avg(list(map(avg, list(knn_adaptive_metrics.training_time.values())))))
    print('Adaptive Avg Test: ',
          avg(list(map(avg, list(knn_adaptive_metrics.test_time.values())))))
