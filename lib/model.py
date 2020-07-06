from lib.metas import MetaModel


class BaseModel(metaclass=MetaModel):

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
