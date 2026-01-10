import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler

def convert(file_path: str, delimiter: str, label_index: int, deduplicate=False):
    """
    :parameter
    file_path (str): file path。
    delimiter (str): file delimiter
    label_index (int): label index

    返回:
    data (numpy.ndarray): feature_matrix
    label (numpy.ndarray): label
    """
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.xlsx':
        data = pd.read_excel(file_path)
    else:
        data = pd.read_csv(file_path, delimiter=delimiter, header = None)

    if deduplicate:
        data = data.drop_duplicates()

    column_name = data.columns
    if label_index == -1:
        label_arr = data.iloc[:, len(column_name) - 1].values
        label_index = len(column_name) - 1
    else:
        label_arr = data.iloc[:, label_index].values

    data_without_label = data.drop(labels = column_name[label_index], axis=1)
    return data_without_label.values, label_arr

def convert_with_enum_feature(file_path: str, delimiter: str, label_index: int, enum_index: list):
    origin_data, origin_label = convert(file_path, delimiter, label_index)
    for index in enum_index:
        if index == label_index: # label is config, convert to number
            enum_label = origin_label
            unique_val = np.unique(enum_label)
            for i in range(len(unique_val)):
                origin_label[enum_label == unique_val[i]] = i
            origin_label = enum_label.astype(np.int64)
        else: # feature is config, convert to number
            enum_feature = origin_data[:, index]
            unique_val = np.unique(enum_feature)
            for i in range(len(unique_val)):
                origin_data[:, index][enum_feature == unique_val[i]] = i
            origin_data = origin_data.astype(np.float64)
    return origin_data, origin_label

def convert_with_std(file_path: str, delimiter: str, label_index: int):
    origin_data, origin_label = convert(file_path, delimiter, label_index)
    scaler = MinMaxScaler()
    std_data = scaler.fit_transform(origin_data)
    return std_data, origin_label

def convert_with_std_deduplicate(file_path: str, delimiter: str, label_index: int, deduplicate=True):
    origin_data, origin_label = convert(file_path, delimiter, label_index, deduplicate=deduplicate)
    scaler = MinMaxScaler()
    std_data = scaler.fit_transform(origin_data)
    return std_data, origin_label

def write(data, true_label, file_name):
    path_prefix = '/Users/dyy/data/dataset/std_dataset/'
    path = path_prefix + file_name + '.csv'
    # convert config label to number
    if not np.issubdtype(true_label.dtype, np.number):
        unique_label = np.unique(true_label)
        for i, l in enumerate(unique_label):
            index = np.where(true_label == l)[0]
            true_label[index] = i
    # convert config data to number
    for col in range(data.shape[1]):
        col_data = data[:, col]
        if not np.issubdtype(col_data.dtype, np.number):
            unique_val = np.unique(col_data)
            for i, l in enumerate(unique_val):
                index = np.where(col_data == l)[0]
                col_data[index] = i

    label_resize = true_label.reshape(-1, 1)
    data_set = np.hstack((label_resize, data))
    np.savetxt(path, data_set, delimiter=',')

def map_label(label):
    unique_label = np.unique(label)
    for i, l in enumerate(unique_label):
        label[label == l] = i
    return label

def file_to_set_with_separator(filename, separator=",", encoding='utf-8', strip_items=True, skip_empty=True):
    """
    :parameter
    filename:
    separator: file delimiter, default ','
    encoding: default 'utf-8'
    strip_items: del space, default 'True'
    skip_empty: skip space element, default 'True'

    return:
    set: element set
    """
    try:
        with open(filename, 'r', encoding=encoding) as file:
            content = file.read()

            # 使用指定的分隔符分割内容
            items = content.split(separator)

            result_set = set()
            for item in items:
                if strip_items:
                    item = item.strip()

                if skip_empty and not item:
                    continue

                result_set.add(item)

            return result_set

    except FileNotFoundError:
        print(f"file '{filename}' not exist")
        return set()
    except Exception as e:
        print(f"read file has exception: {e}")
        return set()


# 更简洁的版本（如果确定数据结构完全一致）
def combine_algorithm_results_simple(algorithm_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    简化版的合并方法，直接拼接各个算法的DataFrame

    参数:
    algorithm_dfs: 字典，键为算法名称，值为DataFrame

    返回:
    多级索引的DataFrame
    """
    # 沿着列方向拼接各个算法的DataFrame
    combined = pd.concat(algorithm_dfs, axis=1)

    # 重命名多级列名
    combined.columns = combined.columns.droplevel(1)

    return combined

def file_exist(path, file_name):
    files = os.listdir(path)
    return files.__contains__(file_name)


def combine_algorithm_results_direct(algorithm_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    直接构建多级索引的最高效方法
    """
    first_df = list(algorithm_dfs.values())[0]
    datasets = first_df.index
    metrics = first_df.columns

    # 创建多级索引
    multi_index = pd.MultiIndex.from_product([datasets, metrics], names=['Dataset', 'Metric'])

    # 创建结果DataFrame
    result_df = pd.DataFrame(index=multi_index, columns=list(algorithm_dfs.keys()))

    # 填充数据
    for alg_name, df in algorithm_dfs.items():
        for i, dataset in enumerate(datasets):
            for j, metric in enumerate(metrics):
                metric_data = df.loc[dataset, metric]
                result_df.loc[(dataset, metric), alg_name] = metric_data

    return result_df


def highlight_optimized_values_corrected(df, metric_rules=None, default_rule='max'):
    """
    根据不同的指标规则高亮每行的最优值（修正版本）

    参数:
    df: 多级索引的DataFrame
    metric_rules: 字典，指定每个指标应该高亮最大值还是最小值
                格式: {'Metric_Name': 'max' 或 'min'}
    default_rule: 默认规则，对于未指定的指标使用此规则

    返回:
    带有样式设置的Styler对象
    """
    if metric_rules is None:
        metric_rules = {}

    def highlight_by_metric(row):
        """根据每行对应的指标规则高亮最优值"""
        # 获取当前行对应的指标名称
        if isinstance(df.index, pd.MultiIndex) and 'Metric' in df.index.names:
            metric_index = df.index.names.index('Metric')
            metric_name = row.name[metric_index]  # 获取指标名称
        else:
            metric_name = 'default'

        # 确定使用哪种规则
        rule = metric_rules.get(metric_name, default_rule)

        if rule == 'max':
            is_optimized = row == row.max()
        else:
            is_optimized = row == row.min()
        style = 'font-weight: bold; background-color: #E8F5E8; color: #006400;'  # 绿色背景表示最大值
        return [style if v else '' for v in is_optimized]

    # 一次性应用所有规则
    styler = df.style.apply(highlight_by_metric, axis=1)

    return styler

def highlight_smart_optimized(df, max_keywords=None, min_keywords=None):
    """
    智能高亮：根据指标名称中的关键词自动判断规则

    参数:
    df: 多级索引的DataFrame
    max_keywords: 表示应该取最大值的指标关键词
    min_keywords: 表示应该取最小值的指标关键词
    """
    if max_keywords is None:
        max_keywords = ['NMI', 'ARI', 'precision', 'recall', 'f1', 'score', 'auc']
    if min_keywords is None:
        min_keywords = ['Time', 'duration', 'latency', 'memory', 'cost', 'error', 'loss']

    # 获取所有指标名称
    if isinstance(df.index, pd.MultiIndex) and 'Metric' in df.index.names:
        metrics = df.index.get_level_values('Metric').unique()
    else:
        metrics = ['Unknown']

    # 为每个指标确定规则
    metric_rules = {}
    for metric in metrics:
        if any(keyword in metric for keyword in max_keywords):
            metric_rules[metric] = 'max'
        elif any(keyword in metric for keyword in min_keywords):
            metric_rules[metric] = 'min'
        else:
            metric_rules[metric] = 'max'  # 默认取最大值

    print(f"自动检测的指标规则: {metric_rules}")

    return highlight_optimized_values_corrected(df, metric_rules)

def mat_data_to_std(file_path, dataset_name, data_key, label_key):
    from scipy.io import loadmat
    full_path = file_path + "/" + dataset_name
    data = loadmat(full_path)
    feature = data.get(data_key)
    labels = data.get(label_key)
    # index zero are labels, other are features
    return feature, labels

def calculate_avg(df):
    columns = df.columns
    for col in columns:
        df.loc["average", col] = df[col].mean()

def load_image_dataset(image_path, target_size=None, flatten=False):
    images = []
    labels = []
    img_list = os.listdir(image_path)
    for img_name in img_list:
        img = cv2.imread(image_path + "/" + img_name, cv2.IMREAD_GRAYSCALE)
        if target_size is not None:
            img = cv2.resize(img, target_size)
        if flatten:
            img = img.flatten()
        img = img.astype(np.int64)
        images.append(img)
        labels.append(int(img_name.split(".")[0].split("_")[2]))
    return np.array(images, dtype=np.int64), np.array(labels, dtype=np.int64)

def save_img_dataset(save_folder, data, label, target_size=(256, 256)):
    for idx in range(data.shape[0]):
        # -------- 步骤1：还原并处理图像矩阵 --------
        # 一维数组 → 16×16二维矩阵
        img_16x16 = data[idx].reshape(target_size)

        # 转为OpenCV兼容的uint8格式（[0,255]）
        if img_16x16.dtype != np.uint8:
            if img_16x16.max() <= 1.0:
                img_16x16 = ((img_16x16 + 1) * 255/2).astype(np.uint8)  # 归一化值转255
            else:
                img_16x16 = img_16x16.astype(np.uint8)  # 其他类型直接转换

        # -------- 步骤2：定义文件名（含索引+标签，便于追溯） --------
        img_name = f"usps_{idx}_{label[idx]}.png"

        # 完整保存路径
        save_path = os.path.join(save_folder, img_name)

        # -------- 步骤3：保存图像 --------
        # cv2.imwrite支持uint8灰度图直接保存，PNG无损格式
        cv2.imwrite(save_path, img_16x16)

        # 可选：打印进度（每100个样本打印一次）
        if (idx + 1) % 100 == 0:
            print(f"save {idx + 1}/{data.shape[0]} simples")

    print(f"save img completed！path：{os.path.abspath(save_path)}")

# 使用示例
if __name__ == "__main__":
    # convert file to std format
    # file_path = '/Users/dyy/data/dataset/uciDataset/'
    # file_name = 'tic-tac-toe_0.data'
    # data, label = convert(file_path+file_name, ",", -1)
    # print(data)
    # print(label)
    # write_file_name = file_name.split('.')[0]
    # write(data, label, write_file_name)

    # statistic exp result
    # exp_dir = "../resource/result/"
    # file_list = os.listdir(exp_dir)
    # exp_dict = {}
    # for file in file_list:
    #     if file in ["实验结果汇总.xlsx", ".DS_Store",
    #                 "mknnDensityDecreaseClusterV2ExpResult.xlsx", "mknnWDensityDecreaseClusterV2ExpResult.xlsx",
    #                 "DensityDecreaseClusterV2ExpResult.xlsx", "combine_exp.xlsx", "combine_exp_v2.xlsx"]:
    #         continue
    #     print(f"file name:{file}")
    #     exp_data = pd.read_excel(exp_dir+file, index_col=0)
    #     alg_name = file.split("Exp")[0]
    #     exp_data = exp_data[['NMI', 'ARI']]
    #     for index in exp_data.index:
    #         if index in ['pathbased', 'segmentation', 'zoo']:
    #             exp_data.drop(index, inplace=True)
    #     calculate_avg(exp_data)
    #     exp_dict[alg_name] = exp_data
    #     # print(exp_data)
    # combine_res = combine_algorithm_results_direct(exp_dict)
    # print(combine_res)
    # res = highlight_smart_optimized(combine_res)
    # print(res)
    # res.to_excel(exp_dir + "combine_exp_v3.xlsx")
    import nn_search, distance
    # img_data, labels = load_image_dataset("/Users/dyy/data/dataset/imgDataset/USPS_img_test", flatten=True)
    # print(img_data.dtype)
    # print(img_data[0].dtype)
    # write(img_data, labels, "usps_t")
    train, train_label = load_image_dataset("/Users/dyy/data/dataset/imgDataset/USPS_img_train", flatten=True)
    test, test_label = load_image_dataset("/Users/dyy/data/dataset/imgDataset/USPS_img_test", flatten=True)
    all_data = np.vstack([train, test])
    distance_matrix = distance.compute_ssim_distance_matrix(all_data, (16, 16), 7, 255)
    np.save("/Users/dyy/data/dataset/imgDataset/usps_distance_all.npy", distance_matrix)
    # dist, index = nn_search.knn_search_remove_self(3, distance_matrix, "precomputed")
    # print(index)
    # print(dist)