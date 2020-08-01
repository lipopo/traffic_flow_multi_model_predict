import numpy as np

from config import time_range
from data import XLSXReader, ExtractData
from lib import TimeGroup, TimeSplit, DataPipe, SeriesDataPipe
from loss import MseLoss
from plot import Line, Wireframe
from model import Svr, Lssvr, BP, GaBP, GaSvr, GaLssvr, GaKnnLssvr
from tasks.base_task import BaseTask


class PlotTask(BaseTask):
    task_name = "plot"
    __doc__ = "\n====绘图任务"
    group = True
    _data = None
    _train_data = None
    _test_data = None

    @property
    def data(self):
        if self._data is None:
            self._data = XLSXReader(
                "asset/30sec.xlsx",
                index_col=0
            ).convert_index_to_datetime()
        return self._data

    @property
    def splited_data(self):
        return self.data.pipe(TimeSplit("2011-05-26 00:00:00"))

    @property
    def train_data(self):
        if self._train_data is None:
            self._train_data = DataPipe(self.splited_data.data[0])
        return self._train_data

    @property
    def test_data(self):
        if self._test_data is None:
            self._test_data = DataPipe(self.splited_data.data[1])
        return self._test_data

    def task_flow(self):
        """绘制交通流量图"""
        data = XLSXReader("asset/30sec.xlsx",
                          index_col=0).convert_index_to_datetime()
        for _time in time_range:
            data.pipe(
                TimeGroup(_time)).pipe(
                lambda _x: _x.mean()).pipe(
                Line(_time))
        Line.finish()
        Line.show()

    def task_param(self):
        """参数精度图"""
        # load data
        data = XLSXReader(
            "asset/30sec.xlsx",
            index_col=0
        ).convert_index_to_datetime().pipe(
            TimeSplit("2011-05-26 00:00:00"))

        # data split
        train_data = DataPipe(data.data[0]).pipe(
            TimeGroup("5Min")
        ).pipe(ExtractData())
        test_data = DataPipe(data.data[1]).pipe(
            TimeGroup("5Min")
        ).pipe(ExtractData())

        # create parameters
        c = np.linspace(0, 1)[1:]
        e = np.linspace(0, 1)[1:]
        C, E = np.meshgrid(c, e)
        logC = np.log(C)
        logE = np.log(E)
        params = np.concatenate(
            [np.expand_dims(i, -1) for i in (C, E)], axis=-1
        ).flatten().reshape(-1, 2)

        # 定义loss
        mse_loss = MseLoss()

        loss_values = []
        # svr model
        for _c, _e in params:
            _model = Svr({"C": _c, "epsilon": _e})

            # fit
            train_data.pipe(
                lambda _x: _model.fit(_x[:, :-1], _x[:, -1]))
            # predict
            data_predict = test_data.pipe(
                lambda _x: _x[:, :-1]).pipe(_model.predict)

            # loss
            preb_value = data_predict.data.get("target_data")
            true_value = test_data.data[:, -1]
            _loss = mse_loss.calc_loss(preb_value, true_value)
            loss_values.append(_loss)
        Loss = np.reshape(np.array(loss_values), logC.shape)
        plot_data = DataPipe([logC, logE, Loss])
        plot_data.pipe(
            Wireframe(
                index=111,
                labels=["$log_2C$", "$log_2\\varepsilon$", "Mse"],
                title="Svr"))
        Wireframe.show()
        Wireframe.show()

    def task_predict(self):
        """分模型流量预测图"""
        # data feature extract
        train_data = self.train_data.pipe(
            TimeGroup("5Min")).pipe(ExtractData())
        train_data_with_cls = self.train_data.pipe(
            TimeGroup("5Min")).pipe(ExtractData(True))
        test_data = self.test_data.pipe(
            TimeGroup("5Min")).pipe(ExtractData())

        print("building models....")
        # train models
        svr_model = Svr()
        lssvr_model = Lssvr()
        lssvr_ga_model = GaLssvr(
            parameter_list=["C", "epsilon"],
            parameter_scaler=[[0, 1], [0, 1]],
            ga_parameter={"max_iter_count": 50}
        )

        bp_model = BP(
            [test_data.data.shape[-1] - 1, 100, 1],
            {"max_iter": 500, "random_state": 1}
        )
        svr_ga_model = GaSvr(
            parameter_list=["C", "epsilon"],
            parameter_scaler=[[0, 1], [0, 1]],
            ga_parameter={"max_iter_count": 50}
        )
        bp_ga_model = GaBP(
            [test_data.data.shape[-1] - 1, 100, 1],
            {"max_iter": 500, "random_state": 1},
            ga_parameter={"max_iter_count": 100}
        )
        ga_knn_lssvr_model = GaKnnLssvr(
            parameter_list=["C", "gamma"],
            parameter_scaler=[[0, 1], [0, 1]],
            ga_parameter={"max_iter_count": 50}
        )

        # load parameters
        svr_ga_model.set_parameter(GaSvr.load_parameter("5Min"))
        lssvr_ga_model.set_parameter(GaLssvr.load_parameter("5Min"))
        bp_ga_model.set_parameter(GaBP.load_parameter("5Min"))
        ga_knn_lssvr_model.set_parameter(GaKnnLssvr.load_parameter("5Min"))

        print("training models...")
        # fit models
        train_data.pipe(lambda _x: svr_model.fit(_x[:, :-1], _x[:, -1]))
        train_data.pipe(
            lambda _x: svr_ga_model.model.fit(_x[:, :-1], _x[:, -1]))
        train_data.pipe(lambda _x: bp_model.fit(_x[:, :-1], _x[:, -1]))
        train_data.pipe(
            lambda _x: bp_ga_model.model.fit(_x[:, :-1], _x[:, -1]))
        train_data.pipe(lambda _x: lssvr_model.fit(_x[:, :-1], _x[:, -1]))
        train_data.pipe(
            lambda _x: lssvr_ga_model.model.fit(_x[:, :-1], _x[:, -1]))
        train_data_with_cls.pipe(
            lambda _x: ga_knn_lssvr_model.fit_knn(_x[:, :-2], _x[:, -2:])
        ).pipe(
            lambda _x: ga_knn_lssvr_model.model.fit(
                np.concatenate(
                    (
                        _x[:, :-2],
                        np.expand_dims(_x[:, -1], -1)),
                    -1), _x[:, -2]))

        print("predicting data...")
        # predict models
        predict_data = test_data.pipe(lambda _x: _x[:, :-1]).pipe(
            svr_model.predict)
        svr_ga_predict_data = test_data.pipe(lambda _x: _x[:, :-1]).pipe(
            svr_ga_model.predict)
        bp_predict_data = test_data.pipe(lambda _x: _x[:, :-1]).pipe(
            bp_model.predict)
        bp_ga_predict_data = test_data.pipe(lambda _x: _x[:, :-1]).pipe(
            bp_ga_model.predict)
        lssvr_predict_data = test_data.pipe(lambda _x: _x[:, :-1]).pipe(
            lssvr_model.predict)
        lssvr_ga_predict_data = test_data.pipe(lambda _x: _x[:, :-1]).pipe(
            lssvr_ga_model.predict)
        ga_knn_lssvr_predict_data = test_data.pipe(lambda _x: _x[:, :-1]).pipe(
            ga_knn_lssvr_model.predict)

        print("plot data")
        # plot predicted
        predict_data.pipe(
            lambda _x: _x.get("target_data")).pipe(
            Line(
                index="211",
                label="svr",
                marker="o",
                title="flow"))
        svr_ga_predict_data.pipe(
            lambda _x: _x.get("target_data")).pipe(
            Line(label="svr_ga", marker="s"))
        bp_predict_data.pipe(
            lambda _x: _x.get("target_data")).pipe(
            Line(label="bp", marker="v"))
        bp_ga_predict_data.pipe(
            lambda _x: _x.get("target_data")).pipe(
            Line(label="bp_ga", marker="8"))
        lssvr_predict_data.pipe(
            lambda _x: _x.get("target_data")).pipe(
            Line(label="lssvr", marker="p"))
        lssvr_ga_predict_data.pipe(
            lambda _x: _x.get("target_data")).pipe(
            Line(label="lssvr_ga", marker="^"))
        ga_knn_lssvr_predict_data.pipe(
            lambda _x: _x.get("target_data")).pipe(
            Line(label="ga_knn_lssvr", marker="x"))
        test_data.pipe(lambda _x: _x[:, -1]).pipe(
            Line(label="true", marker="o"))
        Line.finish()

        preb_data = predict_data.data.get("target_data")
        true_data = test_data.data[:, -1]
        svr_loss_value = DataPipe(
            np.abs(preb_data - true_data))
        svr_loss_value.pipe(Line(
            index="212",
            label="svr",
            marker="o",
            title="flow loss"))

        # bp loss
        bp_preb_data = bp_predict_data.data.get("target_data")
        bp_loss_value = DataPipe(
            np.abs(bp_preb_data - true_data))
        bp_loss_value.pipe(Line(
            index="212",
            label="bp",
            marker="v",
            title="flow loss"))

        # lssvr loss
        lssvr_preb_data = lssvr_predict_data.data.get("target_data")
        lssvr_loss_value = DataPipe(
            np.abs(lssvr_preb_data - true_data))
        lssvr_loss_value.pipe(Line(
            index="212",
            label="lssvr",
            marker="p",
            title="flow loss"))

        # bp_ga loss
        bp_ga_preb_data = bp_ga_predict_data.data.get("target_data")
        bp_ga_loss_value = DataPipe(
            np.abs(bp_ga_preb_data - true_data))
        bp_ga_loss_value.pipe(Line(
            index="212",
            label="bp_ga",
            marker="8",
            title="flow loss"))

        # svr_ga loss
        svr_ga_preb_data = svr_ga_predict_data.data.get("target_data")
        svr_ga_loss_value = DataPipe(
            np.abs(svr_ga_preb_data - true_data))
        svr_ga_loss_value.pipe(Line(
            index="212",
            label="svr_ga",
            marker="s",
            title="flow loss"))
        # lssvr_ga loss
        lssvr_ga_preb_data = lssvr_ga_predict_data.data.get("target_data")
        lssvr_ga_loss_value = DataPipe(
            np.abs(lssvr_ga_preb_data - true_data))
        lssvr_ga_loss_value.pipe(Line(
            index="212",
            label="lssvr_ga",
            marker="^",
            title="flow loss"))
        ga_knn_lssvr_preb_data = ga_knn_lssvr_predict_data.data.get(
            "target_data")
        ga_knn_lssvr_loss_value = DataPipe(
            np.abs(ga_knn_lssvr_preb_data - true_data))
        ga_knn_lssvr_loss_value.pipe(Line(
            index="212",
            label="ga_knn_lssvr",
            marker="x",
            title="flow loss"))
        Line.finish()
        Line.show()

    def task_time_predict(self):
        """分时段预测对比图"""
        markers = ["x", "v", "^"]
        index_map = [311, 312, 313]
        for idx, _time in enumerate(time_range):
            train_data_with_cls = self.train_data.pipe(
                TimeGroup(_time)).pipe(ExtractData(True))
            _test_data = self.test_data.pipe(
                    TimeSplit("2011-05-28 00:00:00")
                ).pipe(lambda _x: _x[1]).pipe(
                TimeGroup(_time)).pipe(
                    ExtractData())
            _index = self.test_data.pipe(TimeGroup(_time)).pipe(
                lambda _x: _x.mean().index).data[-len(_test_data.data):]

            ga_knn_lssvr_model = GaKnnLssvr(
                parameter_list=["C", "gamma"],
                parameter_scaler=[[0, 1], [0, 1]],
                ga_parameter={"max_iter_count": 50}
            )

            # load parameters
            ga_knn_lssvr_model.set_parameter(GaKnnLssvr.load_parameter("5Min"))

            train_data_with_cls.pipe(
                lambda _x: ga_knn_lssvr_model.fit_knn(_x[:, :-2], _x[:, -2:])
            ).pipe(
                lambda _x: ga_knn_lssvr_model.model.fit(
                    np.concatenate(
                        (
                            _x[:, :-2],
                            np.expand_dims(_x[:, -1], -1)),
                        -1), _x[:, -2]))

            ga_knn_lssvr_predict_data = _test_data.pipe(
                lambda _x: _x[:, :-1]).pipe(ga_knn_lssvr_model.predict)
            ga_knn_lssvr_preb_value = ga_knn_lssvr_predict_data.data.get(
                "target_data")
            true_value = _test_data.data[:, -1]

            SeriesDataPipe(
                ga_knn_lssvr_preb_value, index=_index).pipe(
                Line(index=index_map[idx], label="ga-knn-lssvr - " + _time,
                     marker=markers[idx]))
            SeriesDataPipe(true_value, index=_index).pipe(
                Line(label="true - " + _time, marker="8"))
            Line.finish()
        Line.show()
