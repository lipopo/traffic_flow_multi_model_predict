from lib import BaseModel, GA, Individual, Population


class GaKnnLssvm(BaseModel):
    name = "GAKNNLSSVM"
    __doc__ = "基于遗传算法优化的knn_lssvm联合预测模型"

    class ParameterIndividual(Individual):
        @staticmethod
        def rand_feature():
            pass

    ga = GA(Population.generate_population(ParameterIndividual, 100))
