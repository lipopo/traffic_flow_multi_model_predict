from typing import Any, Dict, List

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from lib import BaseModel, GA, Population, ParameterIndividual, LSSVR
from loss import MaeLoss


class GaKnnLssvr(BaseModel):
    name = "GAKNNLSSVR"
    __doc__ = "基于遗传算法优化的knn_lssvm联合预测模型"
    losses = [MaeLoss()]
    use_ga = True
    _ga = None
    _model = None
    _knn_model = None
    _knn_fitted = False

    def __init__(
        self,
        ga_parameter,
        parameter_list,
        parameter_scaler,
        train_kwargs: Dict[str, Any] = {},
        knn_kwargs: Dict[str, Any] = {}
    ):
        """
        @parameter train_kwargs
        @description 训练参数

        @parameter knn_kwargs
        @description knn相关参数

        @parameter ga_parameter
        @description ga参数

        @parameter parameter_list
        @description 参与优化流程的参数名称

        @parameter parameter_scaler
        @description 参数比例尺
        """
        self.train_kwargs = train_kwargs
        self.knn_kwargs = knn_kwargs
        self.ga_parameter = ga_parameter
        self.parameter_scaler = parameter_scaler
        self.parameter_list = parameter_list

    def predict(self, input_data: np.array) -> Dict[str, np.array]:
        """
        @parameter input_data
        @description 输入数据
        """
        # predict with data
        _cls_data = self.knn_model.predict(input_data)
        _predict_data = self.model.predict(np.concatenate(
            (input_data, np.expand_dims(_cls_data, axis=-1)), axis=-1))
        return {
            "input_data": input_data,
            "target_data": _predict_data
        }

    def fit_knn(self, input_data: np.array, target_data: np.array):
        """训练knn模型
        """
        self.knn_model.fit(input_data, target_data[:, -1])
        self._knn_fitted = True

    def model_fit(self, input_data: np.array, target_data: np.array):
        """
        """
        if not self._knn_fitted:
            # fit knn
            self.knn_model.fit(input_data, target_data[:, -1])
            self._knn_fitted = True

        _input_data = np.concatenate(
            (input_data, np.expand_dims(target_data[:, -1], axis=-1)),
            axis=-1)
        # fit lssvr
        self.model.fit(
            _input_data,
            target_data[:, 0])

    def fit(self, input_data: np.array, target_data: np.array):
        """
        @parameter input_data
        @description 输入数据

        @parameter target_data
        @description 目标数据
        """
        self.input_data = input_data
        self.target_data = target_data
        self.model_fit(input_data, target_data)

    def loss(self, parameter) -> List[float]:
        self.set_parameter(parameter)
        # fit first
        self.model_fit(self.input_data, self.target_data)
        preb_value = self.predict(self.input_data).get("target_data")
        true_value = np.expand_dims(self.target_data[:, 1], axis=-1)
        _loss_list = []
        for loss in self.losses:
            _loss_value = loss.calc_loss(preb_value, true_value)
            _loss_list.append(_loss_value)
        return _loss_list

    @property
    def model(self):
        if self._model is None:
            self._model = LSSVR(**self.train_kwargs)
        return self._model

    @property
    def knn_model(self) -> KNeighborsClassifier:
        if self._knn_model is None:
            self._knn_model = KNeighborsClassifier(**self.knn_kwargs)
        return self._knn_model

    @property
    def parameter(self) -> np.array:
        parameters = self.model.get_params()
        parameter_array = []
        for _name, _scaler in zip(
                self.parameter_list, self.parameter_scaler):
            _min, _max = tuple(_scaler)
            _range = _max - _min
            _value = parameters.get(_name)
            parameter_array.append((_value - _min) / _range)
        return np.array(parameter_array)

    def set_parameter(self, parameter):
        _parameter_dict = {}
        idx = 0
        for _name, _scaler in zip(
                self.parameter_list, self.parameter_scaler):
            _min, _max = tuple(_scaler)
            _range = _max - _min
            _value = parameter[idx]
            idx += 1
            _parameter_dict[_name] = _value * _range + _min
        self.model.set_params(**_parameter_dict)

    @property
    def ga(self):
        if self._ga is None:
            self._ga = GA(
                Population.generate_population(
                    ParameterIndividual,
                    25,
                    parameter_size=len(
                        self.parameter),
                    loss_function=self.loss))
        return self._ga
