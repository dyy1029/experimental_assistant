import numpy as np
from base.algorithm_base import AlgorithmBase
from register.alg_injector import AlgInjector

@AlgInjector.AlgBean(alg_name="DBSCAN")
class DBSCAN(AlgorithmBase):
    def __init__(self, features, **kwargs):
        super().__init__()
        self.features = features
        self.center_num = kwargs.get("cluster_num")
        self.labels_ = None

    def fit(self, features = None):
        self.labels_= np.arange(0,self.features.shape[0],1)
        return self

    def predict(self, predict = None):
        return self.labels_
