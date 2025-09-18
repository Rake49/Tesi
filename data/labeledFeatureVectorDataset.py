from typing import List
from .labeledFeatureVector import LabeledFeatureVector
import pandas as pd

class LabeledFeatureVectorDataset:
    def __init__(self, columnsList, targetFeatureName):
        self._columnsName = columnsList
        self._targetFeatureName = targetFeatureName
        self._dataset: List = []
        self._caseIDDominio = set()

    def addLabeledFeatureVector(self, caseId, vector):
        self._dataset.append((caseId, vector))
        self._caseIDDominio.add(caseId)

    def dataset(self):
        return self._dataset
    
    def caseIDDominio(self):
        return self._caseIDDominio
    
    def selectCaseID(self, caseID):
        trace = []
        for id, featureVector in self._dataset:
            if id == caseID:
                trace.append((id, featureVector))
        return trace

    def separateInputFromOutput(self):
        raw_data: list[tuple[List[float], str]] = [
            (labeledFeatureVector.featureVector(), labeledFeatureVector.label())
            for caseID, labeledFeatureVector in self._dataset
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