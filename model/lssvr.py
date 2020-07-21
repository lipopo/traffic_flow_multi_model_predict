from lib import BaseModel, GA, Population, ParameterIndividual


class Lssvr(BaseModel):
    name = "LSSVR"
    __doc__ = "LSSVR预测模型"
    _model = None

    def __init__(self, train_kwargs):
        self.train_kwargs = train_kwargs

    @property
    def model(self):
        if self._model is None:
            self._model = None
        return self._model

    def predict(self, input_data):
        target_data = self.model.predict(input_data)
        return {
            "target_data": target_data,
            "input_data": input_data
        }

    def fit(self, input_data, target_data):
        self.model.fit(input_data, target_data)


class GaLssvr(Lssvr):
    name = "GALSSVR"
    __doc__ = "基于遗传算法优化的LSSVR预测模型"

    @property
    def ga(self):
        if self._ga is None:
            self._ga = GA(
                Population.generate_population(
                    ParameterIndividual,
                    100,
                    loss_function=self.loss,
                    parameter_size=len(
                        self.parameter)))
        return self._ga
