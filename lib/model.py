from collections import namedtuple

import numpy as np

from lib.metas import MetaModel


DataSet = namedtuple('DataSet', ['data', 'target'])


class BaseModel(metaclass=MetaModel):

    losses = []
    test_dataset = None  # 测试数据集

    def snap(self):
        for loss in self.losses:
            loss.snap_point()

    def clear_loss(self):
        for loss in self.losses:
            loss.clear()

    def set_test_dataset(self, test_dataset):
        self.test_dataset = DataSet(**test_dataset)

    @property
    def parameter(self) -> np.array:
        raise NotImplementedError

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

    def save_parameter(self, name: str=""):
        np.save(f"asset/{self.name}_{name}.npy", self.parameter)

    @classmethod
    def load_parameter(cls, name):
        return np.load(f"asset/{cls.name}_{name}.npy")
