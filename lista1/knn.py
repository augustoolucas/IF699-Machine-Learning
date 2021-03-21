import numpy as np

class KNN():
    def __init__(self, K=1, ktype="simple"):
        ktype_list = ["simple", "weight", "adaptive"]
        self.K = K
        self.X_train = None
        self.y_train = None
        
        if ktype not in ktype_list:
            assert False, "Invalid Ktype"
        self.ktype = ktype
    
    def __dist(self, alfa, beta):
        return np.sqrt(np.sum((alfa-beta)**2))
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = np.array(y_train)
        
        if self.ktype == "adaptive":
            self.adaptive_v = np.zeros([len(X_train)]) 
            for i in range(X_train.shape[0]):
                min_rad = 999999
                for j in range(X_train.shape[0]):
                    if i == j or y_train[i] == y_train[j]:
                        continue
                    rad = self.__dist(X_train[i], X_train[j])
                    if rad < min_rad:
                        min_rad = rad
                if min_rad == 999999:
                    print("Warrning! fit min rad")
                self.adaptive_v[i] = min_rad * 0.99999
    
    def __get_pred_class(self, dist):
        if self.ktype == "adaptive":
            dist = dist / (self.adaptive_v + 1e-5)
            
        minIdx     = dist.argsort()[:self.K]
        cand_class = self.y_train[minIdx]
        cand_dist  = dist[minIdx]
        
        if self.ktype == "simple" or self.ktype == "adaptive":
            unique, freq = np.unique(cand_class, return_counts=True)
            return unique[freq.argmax()]
        
        elif self.ktype == "weight":
            unq_class = np.unique(cand_class)
            len_cand_class = len(unq_class)
            w = np.zeros(len_cand_class)
            for i in range(len_cand_class):
                w[i] = np.sum(1.0 / (cand_dist[cand_class == unq_class[i]] + 1e-10))
            return unq_class[w.argmax()]

        else:
            assert False, "Not implemented"
            
    def predict_one(self, test_instance):
        if self.X_train is None or self.y_train is None:
            print("Call fit method before predict")
            return 0
        dist = np.empty(len(self.y_train))
        for idx, instance in enumerate(self.X_train):
            dist[idx] = self.__dist(test_instance, instance)
        
        return self.__get_pred_class(dist)
            
    def predict(self, X_test):
        pred = []
        for test_instance in X_test:
            pred.append( self.predict_one(test_instance) )
        return np.array(pred)
