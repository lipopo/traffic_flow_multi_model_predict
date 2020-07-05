import matplotlib.pyplot as plt

from lib.metas import MetaPlotable


class BasePlotable(metaclass=MetaPlotable):
    index = 111

    def plot(self, data):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        plt.subplot(self.index)
        return self.plot(*args, **kwargs)

    @staticmethod
    def finish():
        plt.legend()

    @staticmethod
    def show():
        plt.show()
