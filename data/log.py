import csv
from typing import Dict

from trace import Trace

class Log:
    def __init__(self, path: str):
        self._log: Dict = {}
        with open(path, mode = 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter = ';')
            for row in reader:
                if row[0] == 'missing_caseid':
                    continue
                if row[0] not in self._log:
                    self._log[row[0]] = Trace(row[0])
                self._log[row[0]].add_event(row)

    def log(self):
        return self._log