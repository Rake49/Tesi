from typing import Dict, List
import pandas as pd
import json
import csv

from .trace import Trace
from .labeledFeatureVector import LabeledFeatureVector

class Log:
    def __init__(self, pathCSV: str, pathFileConf: str):
        self._log: Dict[str, Trace] = {}
        self._dominio = set()
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

    def transformToLabeledFeatureVectorList(self):
        labeledFeatureVectorList: List[LabeledFeatureVector] = []
        for caseID, trace in self._log.items():
            subtraces = trace.subtraces()
            for subtrace in subtraces:
                labeledFeatureVectorList.append(subtrace.transformToLabeledFeatureVector(self._dominio))
        return labeledFeatureVectorList

    def log(self):
        return self._log
    
    def __str__(self):
        out = f""
        for caseID, trace in self._log.items():
            out = out + str(trace)
        return out