from lib import BaseModel, GA, Individual, Population


class Lssvr(BaseModel):
    name = "LSSVR"
    __doc__ = "LSSVR预测模型"


class GaLssvr(Lssvr):
    name = "GALSSVR"
    __doc__ = "基于遗传算法优化的LSSVR预测模型"

    class ParameterIndividual(Individual):
        @staticmethod
        def rand_feature():
            pass

    ga = GA(Population.generate_population(ParameterIndividual, 100))
