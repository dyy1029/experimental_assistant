import enum

CONFIG_ALGORITHM = {}

class ConfigEnum(enum.Enum):
    DEFAULT_ALG_CONFIG_KEY = "default"
    SPECIFIC_ALG_CONFIG_KEY = "specific"

    IS_PARALLEL_KEY = "is_parallel"

    DATASET_ROOT_DIR_KEY = "dataset_root_dir"
    EXPERIMENT_SAVE_DIR_KEY = "result_save_dir"
    DATASET_BLACK_LIST_KEY = "dataset_black_list"
    DATASET_WAITE_LIST_KEY = "dataset_waite_list"
    ALG_SCAN_DIR = "alg_scan_dir"
