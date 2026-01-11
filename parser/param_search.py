import numpy as np

class ParamSearcher:
    def __init__(self, alg_class):
        self.alg_class = alg_class
        self.current_param = None

    def search_best_param_and_result(self, parser, features, label_true, metric_recoder):
        iter_num = 0
        param_combination = parser.get_param_combinations()
        print(f"algorithm {self.alg_class.__name__} parameter space size {len(param_combination)} ")
        for param in param_combination:
            # todo cluster_num config added by algorithm level config to determine
            param['cluster_num'] = len(np.unique(label_true))
            if metric_recoder.record_time:
                metric_recoder.timer_start()
            self.current_param = param
            alg = self.alg_class(features, **param)
            label_pre = alg.fit().predict()
            if metric_recoder.record_time:
                metric_recoder.timer_end()
            metric_recoder.record_best_metric(label_pre, label_true, str(param))
            iter_num += 1
            if iter_num % 20 == 0:
                print(f"algorithm {self.alg_class.__name__} is running {parser.get_config("dataset", "")}, iteration {iter_num} times, percentile {round(iter_num / len(param_combination) * 100, 2)}% ")

    def match_dataset_config(self, dataset, config) -> dict:
        alg_name = self.alg_class.__name__
        alg_default_config = config.get("default")
        dataset_specific_config = config.get("specific").get("dataset")
        if dataset_specific_config is not None and dataset in dataset_specific_config:
            alg_dataset_specific_config = dataset_specific_config.get(dataset)
            if alg_name in alg_dataset_specific_config:
                return alg_dataset_specific_config.get(alg_name)

        if alg_name in alg_default_config:
            return alg_default_config.get(alg_name)
        return {}

    def get_current_param(self):
        return self.current_param