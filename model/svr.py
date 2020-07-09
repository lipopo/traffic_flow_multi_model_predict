from lib import BaseModel, GA, Individual, Population


class Svr(BaseModel):
    name = "SVR"
    __doc__ = "SVR预测模型"

    def predict(self):
        """模型预测结果
        """
        pass

    def fit(self):
        """模型训练
        """
        pass


class GaSvr(Svr):
    name = "GASVR"
    __doc__ = "基于遗传算法优化的SVR预测模型"

    class ParameterIndividual(Individual):
        @staticmethod
        def rand_feature():
            pass

    ga = GA(Population.generate_population(ParameterIndividual, 100))

    def set_parameter(self, parameter):
        """设置模型参数
        """
