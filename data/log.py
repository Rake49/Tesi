import csv
from typing import Dict

from .trace import Trace

class Log:
    def __init__(self, path: str):
        self._log: Dict[str, Trace] = {}
        self._activityCount: Dict[str, Dict] = {}
        with open(path, mode = 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter = ';')
            next(reader, None)
            for row in reader:
                if row[0] == 'missing_caseid':
                    continue
                if row[0] not in self._log:
                    self._log[row[0]] = Trace(row[0])
                self._log[row[0]].add_event(row)
        for caseID, trace in self._log.items():
            self._activityCount[caseID] = trace.activity_count()

    def log(self):
        return self._log
    
    def activity_count(self):
        return self._activityCount
    
    def __str__(self):
        out = f""
        for caseID, trace in self._log.items():
            out = out + str(trace)
        return out