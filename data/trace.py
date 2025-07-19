from typing import List, Dict
from .event import Event

class Trace:
    def __init__(self, caseID: str):
        self._caseID = caseID
        self._events: List[Event] = []
        self._activityCount: Dict = {}

    def caseID(self):
        return self._caseID
    
    def activity_count(self):
        return self._activityCount

    def add_event(self, raw_event: List):
        event = Event(raw_event[1], raw_event[2], raw_event[3])
        self._events.append(event)
        if event.activity() not in self._activityCount:
            self._activityCount[event.activity()] = 0
        self._activityCount[event.activity()] = self._activityCount[event.activity()] + 1

    def __str__(self):
        out = f"CASEID: '{self._caseID}'\n"
        for event in self._events:
            out = out + f"\t{str(event)}\n"
        return out
