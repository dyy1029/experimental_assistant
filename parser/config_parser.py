import itertools
import copy
import numpy as np

class AlgConfigParser:
    def __init__(self, alg_name, dataset, origin_config):
        # algorithm level config
        self.alg_name = alg_name
        self.origin_config = origin_config

        # dataset level config
        self.dataset = dataset
        self.dist_cache_path = None
        self.pre_handle_dataset = None

        # param_combination
        self.all_param_combinations = None

        # do parse config
        self.parse_config()

    def get_param_combinations(self):
        return self.all_param_combinations

    def get_dataset_name(self):
        return self.dataset

    def get_config(self, config_name, default_value):
        if hasattr(self, config_name):
            return getattr(self, config_name)
        return default_value

    def parse_param(self):
        range_param_combinations = None
        fix_param = None
        if "range" in self.origin_config:
            range_param_combinations = self.parse_iteration_param(self.origin_config.get("range"))
        if "fix" in self.origin_config:
            fix_param = self.parse_fix_param(self.origin_config.get("fix"))

        # config contains fix param and range param, combine their
        all_param_combinations = []
        if range_param_combinations is not None and fix_param is not None:
            for range_combination in range_param_combinations:
                for fix_config in fix_param:
                    for param_name, value in fix_config.items():
                        range_combination[param_name] = value
                    all_param_combinations.append(range_combination)
            self.all_param_combinations = all_param_combinations
        else:
            self.all_param_combinations = range_param_combinations if range_param_combinations is not None else fix_param


    def parse_fix_param(self, alg_config):
        if alg_config is not None:
            return [alg_config]
        return None

    def parse_iteration_param(self, config):
        independent_params = {}
        param_name_list = []
        # handle independency params
        for param_name, settings in config.items():
            if "dependency" in settings:
                continue
            sequence = self._generate_sequence(settings)
            independent_params[param_name] = sequence
            param_name_list.append(param_name)
        independent_combination = [] # list element is {param_1:value_1, param_2:value_2,...,param_n:value_n}
        for combination in itertools.product(*independent_params.values()):
            independent_combination.append(dict(zip(independent_params.keys(), combination)))

        # handle dependency params
        full_param_combinations = independent_combination
        for param_name, settings in config.items():
            if "dependency" not in settings:
                continue
            temp_param_combinations = []
            param_name_list.append(param_name)
            for param in independent_combination:
                resolved = self._resolve_dependencies(settings, param)
                sequence = self._generate_sequence(resolved)
                single_param_combination = list(param.values())
                for param_value in sequence:
                    updated_param_value = copy.deepcopy(single_param_combination)
                    updated_param_value.append(param_value)
                    updated_param_combination = dict(zip(param_name_list, updated_param_value))
                    temp_param_combinations.append(updated_param_combination)
            full_param_combinations = temp_param_combinations
        return full_param_combinations

    def _generate_sequence(self, settings):
        """生成参数值序列"""
        start = settings['start']
        end = settings['end']
        step = settings['step']
        sequence = np.arange(start, end, step)
        return sequence

    def _resolve_dependencies(self, settings, dependency_param):
        """解析依赖关系"""
        resolved = {}
        # dependent_sequence = param_sequence.get(settings.get("dependency"))
        dependency_param_name = settings.get("dependency")
        for key, value in settings.items():
            if isinstance(value, str) and value.__contains__(dependency_param_name) and not key.__eq__("dependency"):
                dependency_param_value = dependency_param.get(dependency_param_name)
                context = {dependency_param_name: dependency_param_value}
                # 替换参数引用
                try:
                    resolved[key] = eval(value, context)
                except:
                    error_info = f"parameter config parse error, parameter dependency error, dependency param {dependency_param_name}"
                    raise ValueError(error_info)
            else:
                resolved[key] = value
        return resolved

    def parse_config(self):
        if self.origin_config is None:
            return self
        # compatible that algorithm not need parameter
        if "not_need_param" in self.origin_config:
            self.all_param_combinations = [self.origin_config]
        else:
            if "dist_cache_path" in self.origin_config:
                self.dist_cache_path = self.origin_config.get("dist_cache_path")
            if "pre_handle_dataset" in self.origin_config:
                self.pre_handle_dataset = self.origin_config.get("pre_handle_dataset")
            self.parse_param()
        return self

if __name__ == "__main__":
    import yaml
    config_file = "../resource/config/alg_config.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        error_info = f"config file not found in path {config_file}."
        raise FileNotFoundError(error_info)
    alg_config = config.get('default').get('Dbscan')
    print(alg_config)
    parser = AlgConfigParser("", "Dbscan", alg_config)
    print(parser.all_param_combinations)


