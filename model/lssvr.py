import numpy as np

from lib import BaseModel, GA, Population, ParameterIndividual, LSSVR
from loss import MaeLoss


class Lssvr(BaseModel):
    name = "LSSVR"
    __doc__ = "LSSVR预测模型"
    _model = None
    losses = [MaeLoss()]

    def __init__(self, train_kwargs):
        self.train_kwargs = train_kwargs

    @property
    def model(self):
        if self._model is None:
            self._model = LSSVR(
                **self.train_kwargs)
        return self._model

    def predict(self, input_data):
        """模型预测结果
        """
        target_data = self.model.predict(input_data)
        return {
            "target_data": target_data,
            "input_data": input_data
        }

    def fit(self, input_data, target_data):
        """模型训练
        """
        self.model.fit(input_data, target_data)


class GaLssvr(Lssvr):
    name = "GALSSVR"
    __doc__ = "基于遗传算法优化的LSSVR预测模型"
    use_ga = True
    _ga = None

    def __init__(self, *args, **kwargs):
        self.parameter_list = kwargs.pop("parameter_list")
        self.parameter_scaler = kwargs.pop("parameter_scaler")
        self.ga_parameter = kwargs.pop("ga_parameter")
        super().__init__(*args, **kwargs)

    @property
    def ga(self):
        if self._ga is None:
            self._ga = GA(
                Population.generate_population(
                    ParameterIndividual, 100,
                    parameter_size=len(self.parameter),
                    loss_function=self.loss
                ))
        return self._ga

    @property
    def parameter(self):
        param_dict = self.model.get_params()
        param_list = []
        for param_name, scaler in zip(
                self.parameter_list, self.parameter_scaler):
            parameter_value = param_dict.get(param_name, 0)
            min_val, max_val = tuple(scaler)
            val_range = max_val - min_val
            param_list.append((parameter_value - min_val) / val_range)
        return np.array(param_list)

    def set_parameter(self, parameter):
        """设置指定的参数组合
        """
        parameter_dict = {}
        parameter_scaler = self.parameter_scaler
        for idx, name in enumerate(self.parameter_list):
            min_val, max_val = tuple(parameter_scaler[idx])
            val_range = max_val - min_val
            parameter_dict[name] = parameter[idx] * \
                val_range + min_val
        self.model.set_params(**parameter_dict)

    def loss(self, parameter):
        """计算指参数下的残差
        """
        self.set_parameter(parameter)
        # fit first
        self.model.fit(self.input_data, self.target_data)
        preb_value = self.predict(self.input_data).get("target_data")
        true_value = self.target_data
        _loss_list = []
        for _loss in self.losses:
            _loss_value = _loss.calc_loss(preb_value, true_value)
            _loss_list.append(_loss_value)
        return _loss_list
