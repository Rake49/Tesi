from abc import ABC, abstractmethod

class Classifier(ABC):
    def __init__(self, random_state: int):
        self._random_state = random_state

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def predict(self, featureVectors):
        pass
