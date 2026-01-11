from parser.config_parser import AlgConfigParser
from parser import param_search
import yaml
from util import file_handler
from recorder import metric_evaluate
from config.constants import ConfigEnum
from register.alg_register import AlgRegister
from register.metric_evaluate_register import MetricEvaluateRegister
import os
from concurrent.futures import ThreadPoolExecutor

class ExpAssistant:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config_dict = None
        self.dataset_dir = None
        self.algorithm_list = None
        self.is_parallel = False
        self.dataset_black_list = []
        self.dataset_waite_list = []
        self.metric_list = []
        self.alg_list = []
        self.alg_scan_dir = ""
        self.is_record_time = False
        self.parse_base_config()

    def parse_base_config(self):
        assert self.config_path is not None
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        assert config is not None
        self.config_dict = config

        print(f"algorithm config loaded successfully, \n {self.config_dict}")
        dataset_dir = self.config_dict.get(ConfigEnum.DATASET_ROOT_DIR_KEY.value)
        assert dataset_dir is not None
        self.dataset_dir = dataset_dir

        config_alg_list = list(self.config_dict.get(ConfigEnum.DEFAULT_ALG_CONFIG_KEY.value).keys())
        assert len(config_alg_list) > 0
        self.algorithm_list = config_alg_list

        assert self.config_dict.__contains__(ConfigEnum.METRIC_LIST_KEY.value)
        self.metric_list = self.config_dict.get(ConfigEnum.METRIC_LIST_KEY.value)
        MetricEvaluateRegister.scan_and_inject("", self.metric_list)

        assert self.config_dict.__contains__(ConfigEnum.ALG_LIST_KEY.value)
        self.alg_list = self.config_dict.get(ConfigEnum.ALG_LIST_KEY.value)
        AlgRegister.scan_and_inject(self.alg_scan_dir, self.alg_list)

        if self.config_dict.__contains__(ConfigEnum.IS_PARALLEL_KEY.value):
            self.is_parallel = self.config_dict.get(ConfigEnum.IS_PARALLEL_KEY.value)

        if self.config_dict.__contains__(ConfigEnum.DATASET_BLACK_LIST_KEY.value):
            self.dataset_black_list = self.config_dict.get(ConfigEnum.DATASET_BLACK_LIST_KEY.value)

        if self.config_dict.__contains__(ConfigEnum.DATASET_WAITE_LIST_KEY.value):
            self.dataset_waite_list = self.config_dict.get(ConfigEnum.DATASET_WAITE_LIST_KEY.value)

        if self.config_dict.__contains__(ConfigEnum.ALG_SCAN_DIR.value):
            self.alg_scan_dir = self.config_dict.get(ConfigEnum.ALG_SCAN_DIR.value)

        if self.config_dict.__contains__(ConfigEnum.IS_RECORD_TIME.value):
            self.is_record_time = self.config_dict.get(ConfigEnum.IS_RECORD_TIME.value)


def run():
    assis = ExpAssistant("/Users/dyy/project/python/experimental_assistant/alg_config.yaml")
    if assis.is_parallel:
        print("Parallel mode")
        alg_items = AlgRegister._injector.items()
        with ThreadPoolExecutor(max_workers=len(alg_items)) as executor:
            future_list = []
            for alg_name, alg_class in AlgRegister._injector.items():
                future = executor.submit(run_by_order, alg_name, alg_class, assis)
                future_list.append(future)
            for future in future_list:
                future.result()
    else:
        print("single mode")
        for alg_name, alg_class in AlgRegister._injector.items():
            run_by_order(alg_name, alg_class, assis)

def run_by_order(alg_name, alg_class, assis):
    dataset_list = os.listdir(assis.dataset_dir)
    metric_writer = metric_evaluate.MetricWriterV2(assis.metric_list)
    for dataset in dataset_list:
        # 数据集在黑名单中，跳过
        if dataset in assis.dataset_black_list:
            print(f"dataset {dataset} is in black list")
            continue
        # 数据集有配置白名单，但当前数据集不在白名单中，跳过
        if len(assis.dataset_waite_list) > 0 and dataset not in assis.dataset_waite_list:
            print(f"dataset {dataset} is not in waite list")
            continue
        metric_recoder = metric_evaluate.MetricRecorder(assis.metric_list, MetricEvaluateRegister._injector, True, assis.is_record_time)
        dataset_path = assis.dataset_dir + dataset
        features, label_true = file_handler.convert_with_std_deduplicate(dataset_path, ',', 0)
        alg_searcher = param_search.ParamSearcher(alg_class)
        alg_config = alg_searcher.match_dataset_config(dataset, assis.config_dict)
        parser = AlgConfigParser(alg_name, dataset, alg_config)
        alg_searcher.search_best_param_and_result(parser, features, label_true, metric_recoder)
        metric_writer.append(dataset, metric_recoder.best_metrics)
        write_path = assis.config_dict.get(ConfigEnum.EXPERIMENT_SAVE_DIR_KEY.value) + alg_name + "ExpResult.xlsx"
        metric_writer.to_excel(write_path)

if __name__ == "__main__":
    run()