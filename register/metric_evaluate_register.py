import importlib
import pkgutil
from typing import Dict, Type
from config.constants import CONFIG_METRIC_EVALUATE
from recorder.evaluate import CustomEvaluate

class MetricEvaluateRegister:
    _injector: Dict[str, Type] = {}  # 最终注入的算法类映射

    @classmethod
    def scan_and_inject(cls, package_path: str, metric_list: list[str]):
        """
        扫描指定包下所有带@MetricEvaluate注解的指标计算器，并注入到容器
        :param package_path: 指标计算器类所在包路径（如"recorder"）
        """
        # 清空旧的注册信息
        CONFIG_METRIC_EVALUATE.clear()
        cls._injector.clear()
        # 1. 处理包路径，转为模块路径
        if package_path is not None and package_path != '':
            package_path = package_path.strip("/")
            package_name = package_path.replace("/", ".")
        else:
            package_name = ""
        # 2. 遍历包下所有模块文件
        for _, module_name, is_pkg in pkgutil.walk_packages([package_name]):
            if is_pkg:
                continue  # 跳过子包
            # 动态导入模块（触发@AlgBean注解执行，填充ALG_BEAN_REGISTRY）
            if package_name != "":
                module = importlib.import_module(f"{package_name}.{module_name}")
            else:
                module = importlib.import_module(f"{module_name}")

            for module_name, module_cls in module.__dict__.items():
                if isinstance(module_cls, type) and issubclass(module_cls,CustomEvaluate) and module_cls != CustomEvaluate:
                    CONFIG_METRIC_EVALUATE[module_name] = module_cls
                    print(f"标记指标计算器Bean: {module_name} -> {module_cls.__name__}")

        # 3. 将扫描到的Bean注入到容器
        for metric_name, evaluate in CONFIG_METRIC_EVALUATE.items():
            if metric_name in metric_list:
                cls._injector[metric_name] = evaluate()
        print(f"注入完成！共注入 {len(cls._injector)} 个指标计算器：{list(cls._injector.keys())}")

    @classmethod
    def get_alg_class(cls, metric_name: str) -> Type:
        """获取注入的算法类"""
        if metric_name not in cls._injector:
            raise ValueError(f"指标 {metric_name} 未注入！已注入的指标：{list(cls._injector.keys())}")
        return cls._injector[metric_name]

    @classmethod
    def create_metric_evaluate_instance(cls, alg_name: str):
        """创建注入的算法实例"""
        alg_class = cls.get_alg_class(alg_name)
        return alg_class()

    @classmethod
    def get_all_injected_metric_evaluate(cls) -> list:
        """获取所有已注入的算法名"""
        return list(cls._injector.keys())

    @classmethod
    def get_evaluate(cls, metric_name: str):
        """
        得到metric对应的evaluate
        :param metric_name: 指标名（需与配置文件中的key一致，如NMI、ACC）
        """
        if cls._injector.__contains__(metric_name):
            return cls._injector[metric_name]
        else:
            return None
