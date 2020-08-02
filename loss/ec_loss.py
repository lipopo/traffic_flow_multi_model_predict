import numpy as np

from lib import BaseLoss


class EcLoss(BaseLoss):

    def calc_loss(self, preb_value, true_value):
        """计算EC值
        """
        return 1 - np.sqrt(np.sum((preb_value - true_value) ** 2)) / \
            (np.sqrt(np.sum(preb_value)) + np.sqrt(np.sum(true_value)))
