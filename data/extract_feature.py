import pandas as pd

from lib.feature_extractor import FeatureExtractor


class ExtractData(FeatureExtractor):
    """数据特征提取
    """
    def extract(self, data: pd.DataFrame):
        mean_data = data.mean().iloc[:, 0]  # 均值数据
        var_data = data.var().iloc[:, 0]  # 方差
        diff_data = mean_data.diff()  # 一阶导数
        diff_diff_data = mean_data.diff().diff()  # 二阶导数
        target_data = mean_data.shift(-1)  # 目标值

        # 构建特征表
        feature_data = pd.DataFrame(
            data={
                "mean": mean_data, "var": var_data,
                "diff": diff_data, "diff2": diff_diff_data,
                "target": target_data
            }
        )
        return feature_data.dropna().to_numpy()
