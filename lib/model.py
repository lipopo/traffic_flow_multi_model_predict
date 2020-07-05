from lib.metas import MetaModel


class BaseModel(metaclass=MetaModel):

    def predict(self, input_data):
        raise NotImplementedError
