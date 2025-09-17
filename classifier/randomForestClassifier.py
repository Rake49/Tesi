from .classifier import Classifier
from sklearn.ensemble import RandomForestClassifier as RFC
from typing import override
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class RandomForestClassifier(Classifier):
    def __init__(self, randomState: int):
        super().__init__()
        self._model: RFC = RFC(random_state = randomState, class_weight='balanced')
        self._le = LabelEncoder()

    @override
    def fit(self, dataset):
        x, y = dataset.separateInputFromOutput()
        xdf = dataset.toPandasDF(x)
        yEncoded = self._le.fit_transform(y)
        ys = dataset.toPandasSeries(yEncoded)
        self._model.fit(xdf, ys)

    @override
    def predict(self, dfPredict):
        return self._model.predict(dfPredict)
    
    def model(self):
        return self._model
