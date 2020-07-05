# 任务模块
# 打包主要的命令行指令
import os

from invoke import tasks
import pandas as pd
import matplotlib.pyplot as plt

from data import XLSXReader, ExtractData
from plot import Line
from lib import TimeGroup


@tasks.task
def agg_data(c):
    """合并已经有的数据
    """
    count_data = lambda d: d.count()
    file_list = filter(lambda fname: fname.startswith("2011"), os.listdir("./asset"))
    data = None
    for fname in file_list:
        _d = XLSXReader(
            f"./asset/{fname}", 
            usecols=["交易时间", "WEIGHT_FACT_CASH"], 
            index_col=0).convert_index_to_datetime()
        if not data:
            data = _d
        else:
            data += _d
    data.save("./asset/agg.xlsx")


@tasks.task
def feature_extract(c):
    """特征提取
    """
    assert os.path.exists("asset/agg.xlsx"), "请先运行 inv agg-data 进行数据聚合"
    data = XLSXReader(f"asset/agg.xlsx", index_col=0).convert_index_to_datetime()
    # data.convert_index_to_datetime()
    data_5_min = data.pipe(TimeGroup("5Min")).pipe(ExtractData())
    data_10_min = data.pipe(TimeGroup("10Min")).pipe(ExtractData())
    data_15_min = data.pipe(TimeGroup("15Min")).pipe(ExtractData())

    # save data
    data_5_min.save("asset/5min.xlsx")
    data_10_min.save("asset/10min.xlsx")
    data_15_min.save("asset/15min.xlsx")
