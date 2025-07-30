from .classifier import Classifier
from lightgbm import LGBMClassifier as LGBM
from typing import List
from typing import override
import pandas as pd


class LGBMClassifier(Classifier):
    def __init__(self, random_state: int, dominio):
        super().__init__(random_state)
        self._model: LGBM = LGBM(random_state = random_state)
        self._columnsName = list(dominio)

    def separateInputFromOutput(self, dataset):
        raw_data: list[tuple[List[float], str]] = [
            (labeledFeatureVector.featureVector(), labeledFeatureVector.label())
            for labeledFeatureVector in dataset
        ]
        x = list(map(lambda x: x[0], raw_data))
        y = list(map(lambda x: x[1], raw_data))
        return x, y

    @override
    def fit(self, dataset):
        x, y = self.separateInputFromOutput(dataset)
        xdf = pd.DataFrame(x, columns = self._columnsName)
        ys = pd.Series(y, name = "Label")
        self._model.fit(xdf, ys)

    @override
    def predict(self, featureVectors):
        dfPredict = pd.DataFrame(featureVectors, columns = self._columnsName)
        return self._model.predict(dfPredict)
