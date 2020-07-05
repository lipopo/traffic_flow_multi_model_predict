class FeatureExtractor:
    feature_data = None

    def extract(self, data):
        raise NotImplementedError

    def __call__(self, data):
        self.feature_data = self.extract(data)
        return self.feature_data


class TimeGroup(FeatureExtractor):
    def __init__(self, time_offset):
        self.time_offset = time_offset

    def extract(self, data):
        return data.resample(self.time_offset)


class TimeSplit(FeatureExtractor):
    def __init__(self, time_point):
        self.time_point = time_point

    def extract(self, data):
        return [
            data.loc[:self.time_point],
            data.loc[self.time_point:]
        ]
