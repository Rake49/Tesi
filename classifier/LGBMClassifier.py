from .classifier import Classifier
from xgboost import XGBClassifier as XGB
from typing import override
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class LGBMClassifier(Classifier):
    def __init__(self, randomState: int):
        super().__init__()
        self._model: XGB = XGB(random_state = randomState)
        self._le = LabelEncoder()

    @override
    def fit(self, dataset):
        x, y = dataset.separateInputFromOutput()
        xdf = dataset.toPandasDF(x)
        yEncoded = self._le.fit_transform(y)
        classes = np.unique(yEncoded)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=yEncoded)
        sampleWeights = np.array([weights[label] for label in yEncoded])
        ys = dataset.toPandasSeries(yEncoded)
        self._model.fit(xdf, ys, sample_weight=sampleWeights)

    @override
    def predict(self, dfPredict):
        return self._model.predict(dfPredict)
    
    def model(self):
        return self._model
