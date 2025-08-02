from .classifier import Classifier
from lightgbm import LGBMClassifier as LGBM
from typing import override


class LGBMClassifier(Classifier):
    def __init__(self, random_state: int, dominio, targetFeatureName):
        super().__init__(random_state, list(dominio), targetFeatureName)
        self._model: LGBM = LGBM(random_state = random_state)

    @override
    def fit(self, dataset):
        x, y = self.separateInputFromOutput(dataset)
        xdf = self.toPandasDF(x)
        ys = self.toPandasSeries(y)
        self._model.fit(xdf, ys)

    @override
    def predict(self, featureVectors):
        dfPredict = self.toPandasDF(featureVectors)
        return self._model.predict(dfPredict)
    
    def model(self):
        return self._model
