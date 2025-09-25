from .classifier import Classifier
from sklearn.ensemble import RandomForestClassifier as RFC
from typing import override
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class RandomForestClassifier(Classifier):
    def __init__(self, randomState: int, weights):
        super().__init__()
        self._le = LabelEncoder()
        self._weights = weights
        self._model: RFC = RFC(random_state = randomState)

    @override
    def fit(self, dataset):
        x, y = dataset.separateInputFromOutput()
        xdf = dataset.toPandasDF(x)
        yEncoded = self._le.fit_transform(y)
        weights = {}
        for label, weight in self._weights.items():
            weights[self._le.transform([label])[0]] = weight
        sampleWeights = np.array([weights[label] for label in yEncoded])
        ys = dataset.toPandasSeries(yEncoded)
        self._model.fit(xdf, ys, sample_weight=sampleWeights)

    @override
    def predict(self, dfPredict):
        return self._model.predict(dfPredict)
    
    def model(self):
        return self._model
    
    def weights(self):
        return self._weights
    
    def name(self):
        return 'RandomForest'
