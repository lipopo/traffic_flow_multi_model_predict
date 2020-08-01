"""Least Squares Support Vector Regression."""
import gc
import tracemalloc

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import linalg


class LSSVR(BaseEstimator, RegressorMixin):
    _rbf = None
    idxs = None

    def __init__(self, C=1, kernel='linear', gamma=0.1):
        """
        @parameter C float
        @description 松弛系数

        @parameter kernel str
        @description 核参数

        @parameter gamma float
        @description gamma系数
        """
        self.supportVectors = None
        self.supportVectorLabels = None
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.K = None
        self.bias = None
        self.alphas = None

    def set_params(self, **parameters):
        """
        @parameter **parameters dict
        @description 参数词典
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, x_train, y_train):
        """
        @parameter x_train np.array
        @description 训练集

        @parameter y_train np.array
        @description 训练目标数据
        """
        # tracemalloc.start()
        # if isinstance(self.idxs, type(None)):
        #     self.idxs = np.ones(x_train.shape[0], dtype=bool)
        self.idxs = np.zeros(x_train.shape[0], dtype=bool)
        self.idxs[:] = True

        self.supportVectors = x_train[self.idxs, :].astype(np.float32)
        self.supportVectorLabels = y_train[self.idxs].astype(np.float32)

        self.K = self.kernel_func(
            self.kernel,
            x_train,
            self.supportVectors,
            self.gamma)

        OMEGA = self.K
        OMEGA[self.idxs, np.arange(OMEGA.shape[1])] += 1 / self.C

        D = np.zeros(np.array(OMEGA.shape) + 1)

        D[1:, 1:] = OMEGA
        D[0, 1:] += 1
        D[1:, 0] += 1

        n = len(self.supportVectorLabels) + 1
        t = np.zeros(n)

        t[1:n] = self.supportVectorLabels

        # sometimes this function breaks
        try:
            z = linalg.lsmr(D.T, t)[0]
        except BaseException:
            z = np.linalg.pinv(D).T @ t.ravel()
        self.bias = z[0]
        self.alphas = z[1:]
        self.alphas = self.alphas[self.idxs]
        del D, z
        gc.collect()
        # snapshot = tracemalloc.take_snapshot()
        # state = snapshot.statistics('lineno')
        # print("-"*10)
        # for _s in state[:10]:
        #     print(_s)

    def predict(self, x_test):
        K = self.kernel_func(
            self.kernel,
            x_test,
            self.supportVectors,
            self.gamma)
        return (K @ self.alphas) + self.bias

    def kernel_func(self, kernel, u, v, gamma):
        """
        核过程
        """
        if kernel == 'linear':
            k = np.dot(u, v.T)
        elif kernel == 'rbf':
            k = rbf_kernel(u, v, gamma=gamma)
        return k

    def score(self, X, y, sample_weight=None):
        from scipy.stats import pearsonr
        p, _ = pearsonr(y, self.predict(X))
        return p ** 2

    def norm_weights(self):
        """正则化权重值
        """
        A = self.alphas.reshape(-1, 1) @ self.alphas.reshape(-1, 1).T
        W = A @ self.K[self.idxs, :]
        return np.sqrt(np.sum(np.diag(W)))

    # @property
    # def rbf(self):
    #     if not self._rbf:
    #         self._rbf = self.gamma * RBF()
    #     return self._rbf
