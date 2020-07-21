from typing import Dict, List

import numpy as np

from lib import BaseModel, GA, Population, ParameterIndividual
from loss import MaeLoss


class GaKnnLssvm(BaseModel):
    name = "GAKNNLSSVM"
    __doc__ = "基于遗传算法优化的knn_lssvm联合预测模型"
    losses = [MaeLoss()]
    use_ga = True
    _ga = None

    def __init__(self, train_kwargs, ga_parameter, parameter_scaler):
        self.train_kwargs = train_kwargs

    def predict(self, input_data: np.array) -> Dict[str, np.array]:
        pass

    def fit(self, input_data: np.array, target_data: np.array):
        pass

    def set_parameter(self, parameter):
        pass

    def loss(self, parameter) -> List[float]:
        self.set_parameter(parameter)
        preb_value = self.predict(self.input_data)
        true_value = self.target_data
        _loss_list = []
        for loss in self.losses:
            _loss_value = loss(preb_value, true_value)
            _loss_list.append(_loss_value)
        return _loss_list

    @property
    def parameter(self) -> np.array:
        pass

    @property
    def ga(self):
        if self._ga is None:
            self._ga = GA(
                Population.generate_population(
                    ParameterIndividual,
                    100,
                    parameter_size=len(
                        self.parameter),
                    loss_function=self.loss))
        return self._ga
