class FeatureExtractor:
    feature_data = None

    def extract(self, data):
        raise NotImplementedError

    def __call__(self, data):
        self.feature_data = self.extract(data)
        return self.feature_data
