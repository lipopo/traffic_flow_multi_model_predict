class Pipe:
    data = None
    def __init__(self):
        pass

    def pipe(self, func):
        data = func(self.data)
        if data is not None:
            self.data = data
        return self

    def __add__(self, right_value):
        self.concate(right_value.data)
        return self
