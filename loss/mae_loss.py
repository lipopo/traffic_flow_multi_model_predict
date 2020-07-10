import numpy as np

from lib import BaseLoss


class MaeLoss(BaseLoss):

    def calc_loss(self):
        """计算残差
        """
        return np.mean(self.preb_value - self.true_value)
