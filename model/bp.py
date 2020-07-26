import random
from typing import Dict, Any, Tuple

from sklearn.neural_network import MLPRegressor
import numpy as np

from lib import BaseModel, GA, Individual, Population
from loss import MaeLoss


class BP(BaseModel):
    name = "BP"
    __doc__ = "基于BP神经网络的预测模型"
    _model = None  # 模型
    losses = [MaeLoss()]

    def __init__(
            self,
            layer_sizes: Tuple[int],
            train_kwargs: Dict[str, Any] = {}
            ):
        """bp模型初始化
        @parameter layer_sizes Tuple[int] 网络尺寸
        """
        # setup layers size info
        self.layer_units = layer_sizes
        self.input_layer_size = layer_sizes[1]
        self.hidden_layer_sizes = layer_sizes[1:-1]
        self.output_layer_size = layer_sizes[-1]
        self.train_epochs = train_kwargs.pop("epochs", 100)
        self.train_kwargs = train_kwargs

    @property
    def parameter(self) -> Dict[str, Any]:
        """默认参数
        一般为dict类型
        """
        weights = [cofe.flatten() for cofe in self.model.coefs_]
        bias = self.model.intercepts_
        return np.concatenate(weights + bias)

    @property
    def model(self) -> MLPRegressor:
        """模型
        使用反向传播算法的多层感知器
        """
        if not self._model:
            self._model = MLPRegressor(**self.random_parameter())
            self._model._random_state = np.random
            # 参数初始化
            self._model._initialize(
                np.random.rand(1, self.output_layer_size),
                self.layer_units
            )
        return self._model

    def random_parameter(self):
        """随机初始参数
        关键在于参数范围和参数尺寸
        """
        parameter_dict = {
           'hidden_layer_sizes': self.hidden_layer_sizes
        }
        parameter_dict.update(self.train_kwargs)
        return parameter_dict

    def set_parameter(self, parameter):
        weight_idx = 0
        bias_idx = -sum(self.layer_units)
        coefs = []
        bias = []
        for layer_idx, unit in enumerate(self.layer_units[:-1]):
            coefs.append(
                np.array(parameter[
                    weight_idx:
                    (weight_idx + unit * self.layer_units[layer_idx + 1])
                ]).reshape(
                    (self.layer_units[layer_idx],
                        self.layer_units[layer_idx + 1]))
            )
            bias.append(
                np.array(parameter[
                    bias_idx: bias_idx + self.layer_units[layer_idx + 1]
                ])
            )

        self.model.coefs_ = coefs
        self.model.intercepts_ = bias

    def predict(self, input_data) -> Dict[str, Any]:
        data = {}
        results = self.model.predict(input_data)
        data['target'] = results
        data['input'] = input_data
        return data

    def fit(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data
        for epoch in range(self.train_epochs):
            self.model.fit(input_data, target_data)
            self.snap()

    def loss(self, parameter):
        """计算指定参数下的残差
        """
        self.set_parameter(parameter)
        preb_value = self.predict(self.input_data).get("target")
        true_value = self.target_data
        _loss_list = []
        for _loss in self.losses:
            _loss_value = _loss.calc_loss(preb_value, true_value)
            _loss_list.append(_loss_value)
        return _loss_list


class ParameterIndividual(Individual):
    @staticmethod
    def rand_feature(parameter_size, loss_function):
        parameters = np.random.random(parameter_size)
        return {
            "parameters": parameters,
            "loss_func": loss_function
        }

    @property
    def loss_function(self):
        return self.feature.get("loss_func")

    @property
    def parameters(self):
        return self.feature.get("parameters")

    def calc_fitness(self):
        """计算适应度
        """
        loss_values = self.loss_function(self.parameters)
        # indivdual handler their own fitness and the model
        # handler the loss calc method and others
        return -loss_values[0]

    def crossover(self, other):
        """交叉过程
        """
        parameters = self.feature.get("parameters")
        other_parameters = other.feature.get("parameters")
        for p_idx in range(len(parameters)):
            if random.random() < self.crossover_value:
                parameters[p_idx], other_parameters[p_idx] = \
                        other_parameters[p_idx], parameters[p_idx]
        self.feature["parameters"] = parameters
        other.feature["parameters"] = other_parameters

    def mutation(self, mutation_value):
        """变异过程
        """
        parameters = self.feature.get("parameters")
        for p_idx in range(len(parameters)):
            if random.random() > mutation_value:
                parameters[p_idx] = random.random()
        self.feature["parameters"] = parameters


class GaBP(BP):
    name = "GABP"
    __doc__ = "基于遗传算法优化的BP神经网络模型"
    use_ga = True
    _ga = None  # 缓存ga算法

    def __init__(self, *args, **kwargs):
        self.ga_parameter = kwargs.pop("ga_parameter", {})
        super().__init__(*args, **kwargs)

    @property
    def ga(self):
        if not self._ga:
            self._ga = GA(
                 Population.generate_population(
                     ParameterIndividual,
                     100,
                     parameter_size=len(self.parameter),
                     loss_function=self.loss
                 )
            )
        return self._ga
