import numpy as np

from lib import BaseLoss


class MaeLoss(BaseLoss):

    def calc_loss(self, preb_value, true_value):
        """计算残差
        """
        return np.mean(np.abs(preb_value - true_value))
