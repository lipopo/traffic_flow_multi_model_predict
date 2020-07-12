from copy import deepcopy
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
        self.train_kwargs = train_kwargs

    @property
    def parameter(self) -> Dict[str, Any]:
        """默认参数
        一般为dict类型
        """
        weights = [cofe.flatten() for cofe in self.model.coefs_]
        bias = self.model.intercepts_
        # self.set_test_dataset(test_dataset)
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
        self.model.fit(input_data, target_data)


class ParameterIndividual(Individual):
    def __init__(self, model_parameter):
        self.model = BP(**model_parameter)

    def calc_fitness(self):
        """计算适应度
        """
        set_parameter = self.model.set_parameter
        predict = self.model.predict
        set_parameter(self.feature.get('parameter'))
        preb_target = predict(self.model.test_dataset.data).get('target')
        true_target = self.model.test_dataset.target
        # indivdual handler their own fitness and the model
        # handler the loss calc method and others
        loss = self.model.losses[0](preb_target, true_target).calc_losses()
        return loss

    def crossover(iself, other):
        """交叉过程
        """

    def mutation(iself, mutation_value):
        """变异过程
        """

    def rand_feature(self):
        self.model._model = None  # clear model
        return deep_copy(self)


class GaBP(BP):
    name = "GABP"
    __doc__ = "基于遗传算法优化的BP神经网络模型"
    _ga = None  # 缓存ga算法

    @property
    def ga(self):
        if not self._ga:
            self._ga = GA(
                 Population.generate_population(
                     ParameterIndividual(), 100))
        return self._ga
