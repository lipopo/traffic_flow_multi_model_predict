from lib.feature_extractor import FeatureExtractor


class ExtractData(FeatureExtractor):
    """数据特征提取
    """
    def extract(self, data):
        
        # 统计数据量
        count_data = data.count()

        # 统计其他参数
        mae_data = data.mae()
        
