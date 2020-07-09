from lib.metas import MetaModel


class BaseModel(metaclass=MetaModel):

    def set_parameter(self, parameter):
        """由个体操作，用于设置参数
        """
        pass

    def setup(self):
        """初始化模型
        """
        pass

    def fit(self):
        """训练模型
        """
        raise NotImplementedError

    def predict(self, input_data):
        raise NotImplementedError
