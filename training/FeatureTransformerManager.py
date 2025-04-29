# file: utils/feature_transformer.py
from sklearn.preprocessing import StandardScaler

class FeatureTransformerManager:
    def __init__(self):
        self.transformers = {}

    def fit(self, subject_id, data):
        scaler = StandardScaler()
        self.transformers[subject_id] = scaler.fit(data)
        return self.transformers[subject_id]

    def transform(self, subject_id, data):
        return self.transformers[subject_id].transform(data)

    def fit_global(self, data):
        self.transformers['global'] = StandardScaler().fit(data)
        return self.transformers['global']

    def transform_global(self, data):
        return self.transformers['global'].transform(data)
