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

    features = pd.DataFrame(df.drop(label_column, 1))
    labels = pd.DataFrame(df[label_column])

    return features, labels


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
    features, labels = load_dataset('./kc2.arff')
    label_column = get_data_label(labels)

    e = 1e-12

    features = (features - features.mean()) / (features.max() - features.min() + e)

    k_values = list(range(2, 7, 1))
    
    kf = KFold(n_splits=5)
    kf.get_n_splits(features)
    
    acc_naive_bayes_kmeans = []
    precision_naive_bayes_kmeans = []
    recall_naive_bayes_kmeans = [] 

    acc_naive_bayes = []
    precision_naive_bayes = []
    recall_naive_bayes = [] 
    
    acc_knn = []
    precision_knn = []
    recall_knn = [] 

    for train_idx, test_idx in kf.split(features):
        train_features_bugs = features.iloc[train_idx][labels.iloc[train_idx][label_column] == 1].reset_index(drop=True)
        train_labels_bugs = labels.iloc[train_idx][labels.iloc[train_idx][label_column] == 1].reset_index(drop=True)
        train_features_no_bugs = features.iloc[train_idx][labels.iloc[train_idx][label_column] == 0].reset_index(drop=True)
        train_labels_no_bugs = labels.iloc[train_idx][labels.iloc[train_idx][label_column] == 0].reset_index(drop=True)

        test_features_bugs = features.iloc[test_idx][labels.iloc[test_idx][label_column] == 1].reset_index(drop=True)
        test_labels_bugs = labels.iloc[test_idx][labels.iloc[test_idx][label_column] == 1].reset_index(drop=True)
        test_features_no_bugs = features.iloc[test_idx][labels.iloc[test_idx][label_column] == 0].reset_index(drop=True)
        test_labels_no_bugs = labels.iloc[test_idx][labels.iloc[test_idx][label_column] == 0].reset_index(drop=True)

        train_features = pd.concat([train_features_bugs, train_features_no_bugs]).reset_index(drop=True)
        train_labels = pd.concat([train_labels_bugs, train_labels_no_bugs]).reset_index(drop=True)
        train_labels = np.ravel(train_labels)

        test_features = pd.concat([test_features_bugs, test_features_no_bugs]).reset_index(drop=True)
        test_labels = pd.concat([test_labels_bugs, test_labels_no_bugs]).reset_index(drop=True)
        test_labels = np.ravel(test_labels)

        k_bugs = get_best_k(train_features_bugs, k_values)
        k_no_bugs = get_best_k(train_features_bugs, k_values)

        kmeans_bugs = KMeans(n_clusters=k_bugs).fit(train_features_bugs)
        kmeans_no_bugs = KMeans(n_clusters=k_no_bugs).fit(train_features_no_bugs)

        x_train, y_train = get_train_data((kmeans_bugs, True),
                                          (kmeans_no_bugs, False),
                                          features.columns)

        naive_bayes_kmeans = GaussianNB().fit(x_train, y_train)
        naive_bayes = GaussianNB().fit(train_features, train_labels)
        knn = KNeighborsClassifier(n_neighbors=1).fit(train_features, train_labels)

        prediction = naive_bayes_kmeans.predict(test_features)
        acc_naive_bayes_kmeans.append(metrics.accuracy_score(test_labels, prediction))
        precision_naive_bayes_kmeans.append(metrics.precision_score(test_labels, prediction))
        recall_naive_bayes_kmeans.append(metrics.recall_score(test_labels, prediction))

        prediction = naive_bayes.predict(test_features)
        acc_naive_bayes.append(metrics.accuracy_score(test_labels, prediction))
        precision_naive_bayes.append(metrics.precision_score(test_labels, prediction))
        recall_naive_bayes.append(metrics.recall_score(test_labels, prediction))

        prediction = knn.predict(test_features)
        acc_knn.append(metrics.accuracy_score(test_labels, prediction))
        precision_knn.append(metrics.precision_score(test_labels, prediction))
        recall_knn.append(metrics.recall_score(test_labels, prediction))

    acc_naive_bayes_kmeans = avg(acc_naive_bayes_kmeans)
    precision_naive_bayes_kmeans = avg(precision_naive_bayes_kmeans)
    recall_naive_bayes_kmeans = avg(recall_naive_bayes_kmeans)

    acc_naive_bayes = avg(acc_naive_bayes)
    precision_naive_bayes = avg(precision_naive_bayes)
    recall_naive_bayes = avg(recall_naive_bayes)
    
    acc_knn = avg(acc_knn)
    precision_knn = avg(precision_knn)
    recall_knn = avg(recall_knn)

    print(f'1NN - Acc: {acc_knn}, Precision: {precision_knn}, Recall: {recall_knn}')
    print(f'Naive Bayes - Acc: {acc_naive_bayes}, Precision: {precision_naive_bayes}, Recall: {recall_naive_bayes}')
    print(f'Naive Bayes KMeans - Acc: {acc_naive_bayes_kmeans}, Precision: {precision_naive_bayes_kmeans}, Recall: {recall_naive_bayes_kmeans}')

    plt.figure()
    plt.scatter(['1NN', 'Naive Bayes', 'Naive Bayes Kmeans'], [acc_knn, acc_naive_bayes, acc_naive_bayes_kmeans], label='Avg Accuracy')
    plt.scatter(['1NN', 'Naive Bayes', 'Naive Bayes Kmeans'], [precision_knn, precision_naive_bayes, precision_naive_bayes_kmeans], label='Avg Precision')
    plt.scatter(['1NN', 'Naive Bayes', 'Naive Bayes Kmeans'], [recall_knn, recall_naive_bayes, recall_naive_bayes_kmeans], label='Avg Recall')
    plt.legend()
    plt.show()
