from lib import BaseModel, GA, Individual, Population


class BP(BaseModel):
    name = "BP"
    __doc__ = "基于BP神经网络的预测模型"

    def predict(self, data):
        pass

    def fit(self, data):
        pass


class GaBP(BP):
    name = "GABP"
    __doc__ = "基于遗传算法优化的BP神经网络模型"

    class ParameterIndividual(Individual):
        pass

    ga = GA(Population.generate_population(ParameterIndividual, 100))
