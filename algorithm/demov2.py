import numpy as np
from algorithm.algorithm_base import AlgorithmBase

class DBSCAN(AlgorithmBase):
    def __init__(self, features, **kwargs):
        super().__init__()
        self.features = features
        self.center_num = kwargs.get("cluster_num")
        self.labels_ = None

    def fit(self, features = None):
        self.labels_= np.random.randint(low=1, high=5, size=(self.features.shape[0],))
        return self

    def predict(self, predict = None):
        return self.labels_
