from typing import Dict, List
import json
import csv
import numpy as np

from .trace import Trace
from .labeledFeatureVector import LabeledFeatureVector

class Log:
    def __init__(self, pathCSV: str = None, pathFileConf: str = None):
        self._log: Dict[str, Trace] = {}
        self._dominio = set()
        if pathCSV == None and pathFileConf == None:
            return
        # csv = pd.read_csv(pathCSV, delimiter = ";", skiprows = 1, header = None)
        with open(pathCSV, mode = 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter = ';')
            next(reader, None)
            with open(pathFileConf, 'r') as f:
                config = json.load(f)
            indiciColonne = config.get("indici_colonne_interessate")
            for row in reader:
                if row[indiciColonne[0]] == 'missing_caseid':
                    continue
                if row[indiciColonne[0]] not in self._log:
                    self._log[row[indiciColonne[0]]] = Trace(row[indiciColonne[3]])
                self._dominio.add(row[indiciColonne[1]])
                self._log[row[indiciColonne[0]]].add_event(row[indiciColonne[1]], row[indiciColonne[2]])

    def sortLog(self):
        self._log = {key: value for key, value in sorted(self._log.items(), key = lambda item: item[1].firstItemTimestamp())}

    def _addTrace(self, caseID: str, trace: Trace):
        self._log[caseID] = trace

    def setDominio(self, dominio):
        self._dominio = dominio

    def dominio(self):
        return self._dominio

    def split(self, randomState: int, testSize: float):
        trainSet = Log()
        testSet = Log()
        trainSet.setDominio(self._dominio)
        testSet.setDominio(self._dominio)
        caseIDDomain: Dict[int, str] = {}
        labelCount = {'regular': 0, 'deviant': 0}
        self.sortLog()
        i = 0
        for caseID, trace in self._log.items():
            labelCount[trace.label()] += 1
            caseIDDomain[i] = caseID
            i += 1
        for key, value in labelCount.items():
            labelCount[key] = max(1, value * testSize)
        rng = np.random.default_rng(randomState)
        permutation = rng.permutation(len(self._log))
        for i in permutation:
            caseID = caseIDDomain[i]
            trace = self._log[caseID]
            label = trace.label()
            if labelCount[label] > 0:
                testSet._addTrace(caseID, trace)
                labelCount[label] -= 1
            else:
                trainSet._addTrace(caseID, trace)
        trainSet.sortLog()
        testSet.sortLog()
        return trainSet, testSet

    def transformToLabeledFeatureVectorList(self):
        labeledFeatureVectorList: List[LabeledFeatureVector] = []
        for caseID, trace in self._log.items():
            subtraces = trace.subtraces()
            for subtrace in subtraces:
                labeledFeatureVectorList.append(subtrace.transformToLabeledFeatureVector(self._dominio))
        return labeledFeatureVectorList
    
    def __str__(self):
        out = f""
        for caseID, trace in self._log.items():
            out = out + f"{caseID}:\n" + str(trace)
        return out