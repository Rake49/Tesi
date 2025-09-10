from .classifier import Classifier
from sklearn.ensemble import RandomForestClassifier as RFC
from typing import override
from sklearn.preprocessing import LabelEncoder


class RandomForestClassifier(Classifier):
    def __init__(self, randomState: int):
        super().__init__()
        self._model: RFC = RFC(random_state = randomState)
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
