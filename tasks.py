# 任务模块
import os
import warnings


warnings.simplefilter('ignore')


from invoke import tasks

from data import XLSXReader, ExtractData
from lib import TimeGroup, TimeSplit, BaseModel, MetaModel
from model import BP


@tasks.task
def agg_data(c):
    """合并已经有的数据
    """
    file_list = filter(
            lambda fname: fname.startswith("2011"),
            os.listdir("./asset"))
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
    data = XLSXReader(
            "asset/agg.xlsx",
            index_col=0).convert_index_to_datetime()

    # data.convert_index_to_datetime()
    data_30_sec = data.pipe(TimeGroup("30s")).pipe(ExtractData())
    data_5_min = data.pipe(TimeGroup("5Min")).pipe(ExtractData())
    data_10_min = data.pipe(TimeGroup("10Min")).pipe(ExtractData())
    data_15_min = data.pipe(TimeGroup("15Min")).pipe(ExtractData())

    # save data
    data_30_sec.save("asset/30sec.xlsx")
    data_5_min.save("asset/5min.xlsx")
    data_10_min.save("asset/10min.xlsx")
    data_15_min.save("asset/15min.xlsx")


@tasks.task
def list_models(self):
    """查询可用的模型列表
    """
    models = __import__("model")
    print("当前可用的预测模型有: ")
    for model in models.__dict__.values():
        if type(model) == MetaModel and issubclass(model, BaseModel):
            print(f"    {model.name}-{model.__doc__}")


@tasks.task
def run_model_strategy(self, model_name):
    """运行策略，对于输入数据进行预测
    """
    # load data first
    data_base = XLSXReader(
            "asset/30sec.xlsx",
            index_col=0).convert_index_to_datetime()
    # split data
    data = data_base.pipe(TimeSplit('2011-05-26 00:00:00')).data

    # 构建模型
    # model = BP(layer_sizes=(5, 10, 1))
    # model = BP(layer_sizes=(5, 10, 1))
    # print(model, model.parameter)


@tasks.task()
def fit_model(self, model_name, train_data, test_data, save=False):
    """训练模型
    """
