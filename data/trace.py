from typing import List, Dict
from event import Event

class Trace:
    def __init__(self, caseID: str):
        self._caseID = caseID
        self._events: List = []
        self._activityCount: Dict = {}

    def caseID(self):
        return self._caseID

    def add_event(self, raw_event: List):
        event = Event(raw_event[1], raw_event[2], raw_event[3])
        self._events.append(event)
        if event.activity() not in self._activityCount:
            self._activityCount[event.activity()] = 0
        self._activityCount[event.activity()] = self._activityCount[event.activity()] + 1
