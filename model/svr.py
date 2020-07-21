from sklearn.svm import SVR
import numpy as np

from loss import MaeLoss
from lib import BaseModel, GA, Individual, Population


class Svr(BaseModel):
    name = "SVR"
    __doc__ = "SVR预测模型"
    _model = None
    losses = [MaeLoss()]

    def __init__(self, train_kwargs, parameter_list):
        self.train_kwargs = train_kwargs
        self.parameter_list = parameter_list

    @property
    def model(self):
        if self._model is None:
            self._model = SVR(**self.train_kwargs)
        return self._model

    @property
    def parameter(self):
        param_dict = self.model.get_params()
        param_list = []
        for param_name in self.parameter_list:
            param_list.append(param_dict.get(param_name, None))
        return np.array(param_list)

    def set_parameter(self, parameter):
        """设置指定的参数组合
        """
        parameter_dict = {}
        for idx, name in enumerate(self.paramter_name):
            parameter_dict[name] = parameter[idx]
        self.model.set_params(**parameter_dict)

    def predict(self, input_data):
        """模型预测结果
        """
        target = self.model.predict(input_data)
        return {
            "input_data": input_data,
            "target_data": target
        }

    def fit(self, input_data, target_data):
        """模型训练
        """
        self.model.fit(input_data, target_data)

    def loss(self, parameter):
        """计算指参数下的残差
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
    def loss_funcion(self):
        return self.feature.get("loss_funcion")

    @property
    def parameters(self):
        return self.feature.get("parameters")

    def calc_fitness(self):
        """计算适应度"""
        pass

    def corssover(self, other):
        """交叉"""
        pass

    def mutation(self, mutation_value):
        """变异"""
        pass


class GaSvr(Svr):
    name = "GASVR"
    __doc__ = "基于遗传算法优化的SVR预测模型"
    use_ga = True
    _ga = None

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
