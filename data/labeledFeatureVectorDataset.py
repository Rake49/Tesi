from typing import List
from .labeledFeatureVector import LabeledFeatureVector
import pandas as pd

class LabeledFeatureVectorDataset:
    def __init__(self, columnsList, targetFeatureName):
        self._columnsName = columnsList
        self._targetFeatureName = targetFeatureName
        self._dataset: List[LabeledFeatureVector] = []

    def addLabeledFeatureVector(self, vector):
        self._dataset.append(vector)

    def dataset(self):
        return self._dataset

    def separateInputFromOutput(self):
        raw_data: list[tuple[List[float], str]] = [
            (labeledFeatureVector.featureVector(), labeledFeatureVector.label())
            for labeledFeatureVector in self._dataset
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
    
    def targetFeatureName(self):
        return self._targetFeatureName