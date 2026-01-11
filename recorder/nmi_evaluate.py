from recorder.evaluate import CustomEvaluate
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

class NMI(CustomEvaluate):
    def __init__(self):
        super().__init__()
        self.metric_name = "NMI"

    def metric_value(self, labels_true: np.ndarray, labels_pre: np.ndarray):
        return normalized_mutual_info_score(labels_true, labels_pre)