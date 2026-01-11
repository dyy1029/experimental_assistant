import importlib
import pkgutil
from typing import Dict, Type
from config.constants import CONFIG_ALGORITHM
from algorithm.algorithm_base import AlgorithmBase

class AlgRegister:
    _injector: Dict[str, Type] = {}  # 最终注入的算法类映射

    @classmethod
    def scan_and_inject(cls, package_path: str, alg_list: list[str]):
        """
        扫描指定包下所有带@AlgBean注解的算法类，并注入到容器
        :param package_path: 算法类所在包路径（如"algorithms"）
        """
        # 清空旧的注册信息
        CONFIG_ALGORITHM.clear()
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
                if isinstance(module_cls, type) and issubclass(module_cls, AlgorithmBase) and module_cls != AlgorithmBase:
                    CONFIG_ALGORITHM[module_name] = module_cls
                    print(f"标记算法Bean: {module_name} -> {module_cls.__name__}")

        # 3. 将扫描到的Bean注入到容器
        for alg_name, alg_class in CONFIG_ALGORITHM.items():
            if alg_name in alg_list:
                cls._injector[alg_name] = alg_class
        print(f"注入完成！共注入 {len(cls._injector)} 个算法：{list(cls._injector.keys())}")

    @classmethod
    def get_alg_class(cls, alg_name: str) -> Type:
        """获取注入的算法类"""
        if alg_name not in cls._injector:
            raise ValueError(f"算法 {alg_name} 未注入！已注入的算法：{list(cls._injector.keys())}")
        return cls._injector[alg_name]

    @classmethod
    def create_alg_instance(cls, alg_name: str):
        """创建注入的算法实例"""
        alg_class = cls.get_alg_class(alg_name)
        return alg_class()

    @classmethod
    def get_all_injected_algs(cls) -> list:
        """获取所有已注入的算法名"""
        return list(cls._injector.keys())
