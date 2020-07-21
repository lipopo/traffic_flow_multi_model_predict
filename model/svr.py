from sklearn.svm import SVR

from loss import MaeLoss
from lib import BaseModel, GA, Individual, Population


class Svr(BaseModel):
    name = "SVR"
    __doc__ = "SVR预测模型"
    _model = None
    losses = [MaeLoss()]

    def __init__(self, train_kwargs):
        self.train_kwargs = train_kwargs

    @property
    def model(self):
        if self._model is None:
            self._model = SVR(**self.train_kwargs)
        return self._model

    @property
    def parameter(self):
        param_dict = self.model.get_params()
        c = param_dict.get("C")
        gamma = param_dict.get("param")
        epsilon = param_dict.get("epsilon")
        return [c, gamma, epsilon]

    def set_parameter(self, parameter):
        c, gamma, epsilon = tuple(parameter)
        self.model.set_params(C=c, gammma=gamma, epsilon=epsilon)

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


class ParameterIndividual(Individual):
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
                    ParameterIndividual, 1000))
        return self._ga
