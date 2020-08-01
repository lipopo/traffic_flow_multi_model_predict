from typing import List

from lib import BasePlotable


class Wireframe(BasePlotable):
    use_3d = True

    def __init__(self, labels: List[str] = None, title: str = None):
        self.labels = labels
        self.title = title

    def plot(self, data):
        X, Y, Z = tuple(data)
        self.axes.plot_wireframe(X, Y, Z)

        if self.labels is not None:
            self.axes.set_xlabel(self.labels[0])
            self.axes.set_ylabel(self.labels[1])
            self.axes.set_zlabel(self.labels[2])

        if self.title is not None:
            self.axes.set_title(self.title)
