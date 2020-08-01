from lib import BasePlotable

import matplotlib.pyplot as plt


class Line(BasePlotable):

    def __init__(
        self,
        label: str = None,
        marker: str = None,
        title: str = None
    ):
        self.label = label
        self.marker = marker
        self.title = title

    def plot(self, data):
        plt.plot(
            data,
            label=self.label,
            marker=self.marker,
        )
