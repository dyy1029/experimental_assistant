import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import datetime

# record the best experiment result of one algorithm in some datasets
class MetricEvaluate:
    def __init__(self, metrics=None, record_param=False, record_time=False):
        self.metrics = self.parse_metrics(metrics, record_param, record_time)
        self.record_param = record_param
        self.best_metrics = {}
        self.iter_num = 0
        self.record_time = record_time
        self.execute_time = {}
        self.current_metric = None

    """
    invoke after function timer_end if you need record algorithm running time
    """
    def record_best_metric(self, label_true, label_pre, param=None):
        self.current_metric = ""
        for metric in self.metrics:
            if metric not in self.metrics:
                print(f"invalid metric {metric}")
            if 'NMI'.__eq__(metric):
                nmi = normalized_mutual_info_score(label_true, label_pre)
                self.current_metric += f'NMI:{nmi} '
                self.best_metrics[metric] = max(self.best_metrics[metric], nmi) if self.best_metrics.__contains__(metric) else nmi
                if self.record_param and self.best_metrics[metric] == nmi:
                    self.best_metrics[metric + "_param"] = param
                if self.record_time and self.best_metrics[metric] == nmi:
                    key = param if param is not None else self.iter_num
                    self.best_metrics[metric + "_time"] = self.execute_time.get(key)
            elif 'ARI'.__eq__(metric):
                ari = adjusted_rand_score(label_true, label_pre)
                self.current_metric += f'ARI:{ari} '
                self.best_metrics[metric] = max(self.best_metrics[metric], ari) if self.best_metrics.__contains__(metric) else ari
                if self.record_param and self.best_metrics[metric] == ari:
                    self.best_metrics[metric + "_param"] = param
                    self.current_metric += f'param:{param} '
                if self.record_time and self.best_metrics[metric] == ari:
                    key = param if param is not None else self.iter_num
                    self.best_metrics[metric + "_time"] = self.execute_time.get(key)
        if self.record_param:
            self.current_metric += f'param:{param} '
        self.iter_num += 1

    def get_best_metric(self):
        return self.best_metrics

    def get_current_metric(self):
        return self.current_metric

    def parse_metrics(self, metrics, record_param, record_time):
        base_metric = ['NMI', 'ARI'] if metrics is None else metrics
        complete_metric = []
        for metric in base_metric:
            complete_metric.append(metric)
            if record_param:
                complete_metric.append(metric + "_param")
            if record_time:
                complete_metric.append(metric + "_time")
        return complete_metric

    def clear(self):
        self.best_metrics = {}
        self.execute_time = {}
        self.iter_num = 0

    """
    invoke before algorithm start
    """
    def timer_start(self, param=None):
        if not self.record_time:
            return
        key = param if param is not None else self.iter_num
        self.execute_time[key] = datetime.datetime.now().timestamp()

    """
    invoke before function record_best_metric
    """
    def timer_end(self, param=None):
        if not self.record_time:
            return
        key = param if param is not None else self.iter_num
        if self.execute_time.__contains__(key):
            self.execute_time[key] = datetime.datetime.now().timestamp() - self.execute_time[key]

class MetricWriter:
    """
    write each dataset metric to excel
    """
    def __init__(self, metrics=None, record_param=False, record_time=False):
        self.metric_evaluate = MetricEvaluate(metrics, record_param, record_time)
        self.df = pd.DataFrame(columns=self.metric_evaluate.metrics)

    def append(self, dataset_name):
        for metric, value in self.metric_evaluate.best_metrics.items():
            self.df.loc[dataset_name, metric] = value

    def to_excel(self, path):
        if path is None or ''.__eq__(path):
            return
        self.df.to_excel(path)

    def get_metric_evaluate(self):
        return self.metric_evaluate

    def record_best_metric(self, label_true, label_pre, param=None):
        self.metric_evaluate.record_best_metric(label_true, label_pre, param)

    def get_best_metric(self):
        return self.metric_evaluate.get_best_metric()

    def get_full_metric_info(self):
        return self.df