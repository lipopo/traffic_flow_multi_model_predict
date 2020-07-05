# 任务模块
# 打包主要的命令行指令
import os

from invoke import tasks
import pandas as pd
import matplotlib.pyplot as plt

from data import XLSXReader
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
        _d = XLSXReader(f"./asset/{fname}", usecols=["交易时间", "WEIGHT_FACT_CASH"], index_col=0)
        if not data:
            data = _d
        else:
            data += _d
    data.save("./asset/agg.xlsx")


@tasks.task
def feature_extract(c):
    """特征提取
    """
