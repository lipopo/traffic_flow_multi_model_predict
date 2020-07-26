from lib import BasePlotable

import matplotlib.pyplot as plt


class Line(BasePlotable):

    def __init__(self, label: str = None):
        self.label = label

    def plot(self, data):
        plt.plot(data, label=self.label)
