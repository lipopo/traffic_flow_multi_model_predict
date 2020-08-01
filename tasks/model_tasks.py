from config import model_config, time_range
from data import XLSXReader, ExtractData
from lib import MetaModel, BaseModel, TimeSplit, DataPipe, TimeGroup
from plot import Line

from tasks.base_task import BaseTask


class ModelTask(BaseTask):
    task_name = "model"
    __doc__ = f"""\n{'=' * 4} 模型任务"""
    group = True
    _models = None

    @property
    def models(self):
        if self._models is None:
            self._models = []
            _models = __import__("model")
            for _model in _models.__dict__.values():
                if isinstance(
                        _model,
                        MetaModel) and issubclass(
                        _model,
                        BaseModel):
                    self._models.append(_model)
        return self._models

    def get_model(self, model_name):
        for model in self.models:
            if model.name == model_name:
                return model
        return None

    def get_parameter(self, model_name, _time):
        model = self.get_model(model_name)
        return model.load_parameter(_time)

    def task_list(self):
        """列出当前模型"""
        print(
            "当前支持的模型:"
        )
        for model in self.models:
            print(
                " " * 4,
                f"{model.name}-{model.__doc__}"
            )

    def task_fit(self, model_name, with_cls='False'):
        """训练模型 - model.fit <model_name>"""
        # load data first
        data_base = XLSXReader(
            "asset/30sec.xlsx",
            index_col=0).convert_index_to_datetime()
        # split data
        train_data, test_data = tuple(data_base.pipe(
            TimeSplit('2011-05-26 00:00:00')).data)
        train_data_pipe = DataPipe(train_data)  # 训练数据Pipe
        test_data_pipe = DataPipe(test_data)  # 测试数据Pipe

        data = {}
        with_cls = with_cls == "True"
        for _time in time_range:
            data[f"train_{_time}"] = train_data_pipe.pipe(
                TimeGroup(_time)).pipe(ExtractData(with_cls=with_cls))
            data[f"test_{_time}"] = test_data_pipe.pipe(
                TimeGroup(_time)).pipe(ExtractData(with_cls=with_cls))

        Model = self.get_model(model_name)

        if Model is None:
            print(f"{model_name}模型不存在")
            return

        # 构建模型 - 按照不同的时间范围，以及不同的模型构建
        # 训练模型
        for _time in time_range:
            print("Start training at {}".format(_time))
            model = Model(**model_config.get(model_name))
            train_data = data.get(f"train_{_time}")
            train_data.pipe(
                lambda d: model.fit(
                    d[:, : -2 if with_cls else -1],
                    d[:, -2 if with_cls else -1:]))
            model.save_parameter(_time)

    def task_run(self, model_name):
        """运行模型 - model.run <model_name>"""
        data = XLSXReader(
            "./asset/30sec.xlsx",
            index_col=0
        ).convert_index_to_datetime()
        Model = self.get_model(model_name)
        model = Model(**model_config.get(model_name))
        for idx, _time in enumerate(time_range):
            model.set_parameter(self.get_parameter(model_name, _time))
            predict = model.predict
            predict_data = data.pipe(TimeGroup(_time)).pipe(
                ExtractData()).pipe(lambda data: data[:, :-1]).pipe(predict)

            # plot data
            predict_data.pipe(
                lambda d: d['target']).pipe(
                Line(
                    label=f"{_time}_predict",
                    index=len(time_range) *
                    100 +
                    11 +
                    idx))
            data.pipe(TimeGroup(_time)).pipe(ExtractData()).pipe(
                lambda data: data[:, -1]).pipe(Line(label=f"{_time}_true"))
            Line.finish()
        Line.show()
