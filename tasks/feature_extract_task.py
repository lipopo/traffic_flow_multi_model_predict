import os

from data import XLSXReader, PreprocessData
from lib import TimeGroup

from tasks.base_task import BaseTask


class FeatureExtractTask(BaseTask):
    task_name="feature_extract"
    __doc__="""数据预处理"""

    def run(self):
        assert os.path.exists("asset/agg.xlsx"), "请先运行 inv agg-data 进行数据聚合"
        data = XLSXReader(
                "asset/agg.xlsx",
                index_col=0).convert_index_to_datetime()
    
        # data.convert_index_to_datetime()
        data_30_sec = data.pipe(TimeGroup("30s")).pipe(PreprocessData())
        data_5_min = data.pipe(TimeGroup("5Min")).pipe(PreprocessData())
        data_10_min = data.pipe(TimeGroup("10Min")).pipe(PreprocessData())
        data_15_min = data.pipe(TimeGroup("15Min")).pipe(PreprocessData())
    
        # save data
        data_30_sec.save("asset/30sec.xlsx")
        data_5_min.save("asset/5min.xlsx")
        data_10_min.save("asset/10min.xlsx")
        data_15_min.save("asset/15min.xlsx")
