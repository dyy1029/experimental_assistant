from recorder.evaluate import CustomEvaluate
from sklearn.metrics import adjusted_rand_score
import numpy as np

class ARI(CustomEvaluate):
    def __init__(self):
        super().__init__()
        self.metric_name = "ARI"

    def metric_value(self, labels_true: np.ndarray, labels_pre: np.ndarray):
        return adjusted_rand_score(labels_true, labels_pre)