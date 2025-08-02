from abc import ABC, abstractmethod
from typing import List
import pandas as pd

class Classifier(ABC):
    def __init__(self, columnsList, targetFeatureName):
        self._columnsName = columnsList
        self._targetFeatureName = targetFeatureName

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def predict(self, featureVectors):
        pass

    def separateInputFromOutput(self, dataset):
        raw_data: list[tuple[List[float], str]] = [
            (labeledFeatureVector.featureVector(), labeledFeatureVector.label())
            for labeledFeatureVector in dataset
        ]
        x = list(map(lambda x: x[0], raw_data))
        y = list(map(lambda x: x[1], raw_data))
        return x, y
    
    def toPandasDF(self, data):
        return pd.DataFrame(data, columns = self._columnsName)
    
    def toPandasSeries(self, data):
        return pd.Series(data, name = self._targetFeatureName)
    
    def columnsName(self):
        return self._columnsName