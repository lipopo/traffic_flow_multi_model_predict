import numpy as np
import pandas as pd

from data import XLSXReader, ExtractData
from config import time_range, scaler_config
from lib import TimeGroup, TimeSplit, DataPipe, SeriesDataPipe
from loss import MseLoss, MaeLoss, EcLoss
from model import Svr, Lssvr, BP, GaBP, GaSvr, GaLssvr, GaKnnLssvr
from tasks.base_task import BaseTask


class DataTask(BaseTask):
    task_name = "data"
    __doc__ = "\n====数据任务"
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

    def task_export(self, filename: str):
        """输出精度数据 <filename> 文件名"""
        loss_data = {}
        for _time in time_range:
            # data feature extract
            train_data = self.train_data.pipe(
                TimeGroup(_time)).pipe(ExtractData())
            train_data_with_cls = self.train_data.pipe(
                TimeGroup(_time)).pipe(ExtractData(True))
            test_data = self.test_data.pipe(
                TimeGroup(_time)).pipe(ExtractData())
            _index = self.test_data.pipe(TimeGroup(_time)).pipe(
                lambda _x: _x.mean().index
            ).data[-len(test_data.data):]

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
            ga_knn_lssvr_predict_data = test_data.pipe(
                lambda _x: _x[:, :-1]).pipe(
                ga_knn_lssvr_model.predict)

            # output data
            svr_series = SeriesDataPipe(
                predict_data.data.get('target_data'),
                index=_index
            )
            svr_ga_series = SeriesDataPipe(
                svr_ga_predict_data.data.get('target_data'),
                index=_index
            )
            bp_series = SeriesDataPipe(
                bp_predict_data.data.get('target_data'),
                index=_index
            )
            bp_ga_series = SeriesDataPipe(
                bp_ga_predict_data.data.get('target_data'),
                index=_index
            )
            lssvr_series = SeriesDataPipe(
                lssvr_predict_data.data.get('target_data'),
                index=_index
            )
            lssvr_ga_series = SeriesDataPipe(
                lssvr_ga_predict_data.data.get('target_data'),
                index=_index
            )
            ga_knn_lssvr_series = SeriesDataPipe(
                ga_knn_lssvr_predict_data.data.get('target_data'),
                index=_index
            )
            true_series = SeriesDataPipe(
                test_data.data[:, -1],
                index=_index
            )

            series_data = {
                'svr': svr_series.data,
                'svr_ga': svr_ga_series.data,
                'bp': bp_series.data,
                'bp_ga': bp_ga_series.data,
                'lssvr': lssvr_series.data,
                'lssvr_ga': lssvr_ga_series.data,
                'ga_knn_lssvr': ga_knn_lssvr_series.data,
                'true_data': true_series.data
            }
            data = pd.DataFrame(
                data=series_data,
                index=_index
            ) * scaler_config.get("max")

            data.to_excel("./asset/export-{}-{}.xlsx".format(
                filename, _time
            ))

            losses = {
                "mape": MaeLoss(),
                "rmse": MseLoss(),
                "ec": EcLoss()
            }

            model_data = {
                "svr": svr_series,
                "svr-ga": svr_ga_series,
                "bp": bp_series,
                "bp-ga": bp_ga_series,
                "lssvr": lssvr_series,
                "lssvr-ga": lssvr_ga_series,
                "ga-knn-lssvr": ga_knn_lssvr_series
            }
            loss_data[_time] = {
                _model_name +
                "-" +
                _loss_name: _loss.calc_loss(
                    _model_series.data,
                    true_series.data) for _loss_name,
                _loss in losses.items() for _model_name,
                _model_series in model_data.items()}
        pd.DataFrame(loss_data).to_excel(
            "asset/{}-loss.xlsx".format(filename))
