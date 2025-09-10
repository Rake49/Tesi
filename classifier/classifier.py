from abc import ABC, abstractmethod

class Classifier(ABC):
    def __init__(self):
        self._le = None

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def predict(self, featureVectors):
        pass
    