import matplotlib.pyplot as plt

from lib.metas import MetaPlotable


class BasePlotable(metaclass=MetaPlotable):
    index = 111
    use_3d = False

    def plot(self, data):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if self.use_3d:
            self.fig = plt.figure()
            self.axes = self.fig.add_subplot(self.index, projection='3d')
        else:
            plt.subplot(self.index)
        return self.plot(*args, **kwargs)

    @staticmethod
    def finish():
        plt.legend()

    @staticmethod
    def show():
        plt.show()
