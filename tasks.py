import os
import warnings


warnings.simplefilter('ignore')


from invoke import tasks

from config import time_range, model_config
from data import XLSXReader, ExtractData, PreprocessData
from lib import TimeGroup, TimeSplit, BaseModel, MetaModel, DataPipe
from plot import Line
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
    data_30_sec = data.pipe(TimeGroup("30s")).pipe(PreprocessData())
    data_5_min = data.pipe(TimeGroup("5Min")).pipe(PreprocessData())
    data_10_min = data.pipe(TimeGroup("10Min")).pipe(PreprocessData())
    data_15_min = data.pipe(TimeGroup("15Min")).pipe(PreprocessData())

    # save data
    data_30_sec.save("asset/30sec.xlsx")
    data_5_min.save("asset/5min.xlsx")
    data_10_min.save("asset/10min.xlsx")
    data_15_min.save("asset/15min.xlsx")


def get_model(model_name):
    models = __import__("model")
    for model in models.__dict__.values():
        if type(model) == MetaModel and issubclass(model, BaseModel) and model.name == model_name:
            return model
    return None


def get_parameter(model_name, _time):
    Model = get_model(model_name)
    return Model.load_parameter(_time)


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
    """运行策略
    """
    data = XLSXReader(
            "./asset/30sec.xlsx", 
            index_col=0
        ).convert_index_to_datetime()
    Model = get_model(model_name)
    model = Model(**model_config.get(model_name))
    for idx, _time in enumerate(time_range):
        model.set_parameter(get_parameter(model_name, _time))
        predict = model.predict
        predict_data = data.pipe(TimeGroup(_time)).pipe(ExtractData()).pipe(lambda data: data[:, :-1]).pipe(predict)

        # plot data
        predict_data.pipe(lambda d: d['target']).pipe(Line(label=f"{_time}_predict", index=len(time_range) * 100 + 11 + idx))
        data.pipe(TimeGroup(_time)).pipe(ExtractData()).pipe(lambda data: data[:, -1]).pipe(Line(label=f"{_time}_true"))
        Line.finish()
    Line.show()


@tasks.task()
def fit_model(self, model_name):
    """训练模型
    """
    # load data first
    data_base = XLSXReader(
            "asset/30sec.xlsx",
            index_col=0).convert_index_to_datetime()
    # split data
    train_data, test_data = tuple(data_base.pipe(TimeSplit('2011-05-26 00:00:00')).data)
    train_data_pipe = DataPipe(train_data)  # 训练数据Pipe
    test_data_pipe = DataPipe(test_data)  # 测试数据Pipe

    data = {}
    for _time in time_range:
        data[f"train_{_time}"] = train_data_pipe.pipe(TimeGroup(_time)).pipe(ExtractData())
        data[f"test_{_time}"] = test_data_pipe.pipe(TimeGroup(_time)).pipe(ExtractData())

    Model = get_model(model_name)

    if Model is None:
        c.echo(f"{model_name}模型不存在")
        return

    # 构建模型 - 按照不同的时间范围，以及不同的模型构建
    # 训练模型
    for _time in time_range:
        model = Model(**model_config.get(model_name))
        data.get(f"train_{_time}").pipe(lambda d: model.fit(d[:, :-1], d[:, -1]))
        model.save_parameter(_time)

