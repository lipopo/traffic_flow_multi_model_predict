import numpy as np

from lib import BaseLoss


class MseLoss(BaseLoss):

    def calc_loss(self, preb_value, true_value):
        """计算残差
        """
        return np.mean((preb_value - true_value)**2)
