import numpy as np

class CustomEvaluate:
    def __init__(self):
        self.metric_name = None

    def metric_value(self, labels_true: np.ndarray, labels_pre: np.ndarray):
        pass

    # default metric compare method, the larger value the better
    def compare(self, old_metric_val, new_metric_val):
        if old_metric_val < new_metric_val:
            return -1
        elif old_metric_val > new_metric_val:
            return 1
        else:
            return 0