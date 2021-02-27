import pandas as pd
import sklearn as sk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.io.arff import loadarff

def load_dataset(filename):
    arff_data = loadarff(filename)
    df = pd.DataFrame(arff_data[0])
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.dropna()

    attributes = pd.DataFrame(df.drop('defects', 1))
    labels = pd.DataFrame(df['defects'])

    return attributes, labels


if __name__ == '__main__':
    attributes, labels = load_dataset('./kc1.arff')
    labels = labels.astype(str).replace({"b'false'": 0, "b'true'": 1})

    x_train, x_test, y_train, y_test = train_test_split(attributes, labels,
                                                        test_size=0.2)

    print('Train')
    print(f'False: {len(y_train) - y_train["defects"].sum()}')
    print(f'True: {y_train["defects"].sum()}')

    print('Test:')
    print(f'False: {len(y_test) - y_test["defects"].sum()}')
    print(f'True: {y_test["defects"].sum()}')

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)

    scaler = StandardScaler().fit(x_test)
    x_test = scaler.transform(x_test)

    y_train = y_train['defects'].tolist()
    y_test = y_test['defects'].tolist()

    classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
