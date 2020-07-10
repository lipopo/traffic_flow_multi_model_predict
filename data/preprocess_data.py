from lib.feature_extractor import FeatureExtractor


class PreprocesData(FeatureExtractor):
    """数据预处理
    """
    def extract(self, data):
        # 拿到的是聚合数据，对于聚合数据进行特征提取
        count_data = data.count()

        # 归一化
        max_count = count_data.max()
        min_count = count_data.min()
        count_norm_data = (count_data - min_count) / (max_count - min_count)

        # 返回归一化后的数值
        return count_norm_data
