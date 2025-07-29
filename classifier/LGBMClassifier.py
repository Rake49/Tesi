from .classifier import Classifier
from lightgbm import LGBMClassifier as LGBM
from typing import override
from typing import List


class LGBMClassifier(Classifier):
    def __init__(self, random_state: int):
        super().__init__(random_state)
        self._model: LGBM = LGBM(random_state = random_state)

    @override
    def fit(self, dataset):
        raw_data: list[tuple[List[float], str]] = [
            (labeledFeatureVector.featureVector(), labeledFeatureVector.label())
            for labeledFeatureVector in dataset
        ]
        x = list(map(lambda x: x[0], raw_data))
        y = list(map(lambda x: x[1], raw_data))
        self._model.fit(x, y)

    @override
    def predict(self, featureVector):
        return self._model.predict([featureVector])[0]
