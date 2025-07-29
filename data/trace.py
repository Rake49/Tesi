from typing import List
from .event import Event
from multipledispatch import dispatch
from .labeledFeatureVector import LabeledFeatureVector

class Trace:
    def __init__(self, label: str):
        self._label = label
        self._events: List[Event] = []

    def label(self):
        return self._label

    @dispatch(str, str)
    def add_event(self, activity: str, timestamp: str):
        event = Event(activity, timestamp)
        newPosition = self.indexOfNextItemAfter(event)
        self._events = self._events[0: newPosition] + [event] + self._events[newPosition:]
    
    @dispatch(Event)
    def add_event(self, event: Event):
        self._events.append(event)

    def indexOfNextItemAfter(self, event: Event):
        i = 0
        while (i < len(self._events) and event.timestamp() >= self._events[i].timestamp()):
            i = i + 1
        return i

    def subtraces(self):
        subtraces: List[Trace] = []
        for i in range(0, len(self._events)):
            subtraces.append(Trace(self._label))
            for j in range(0, i + 1):
                subtraces[i].add_event(self._events[j])
        return subtraces


    def transformToLabeledFeatureVector(self, dominio):
        labeledFeatureVector = LabeledFeatureVector(self._label, len(dominio))
        i = 0
        for activity in dominio:
            for event in self._events:
                if event.activity() == activity:
                    labeledFeatureVector.incrementValue(i)
            i = i + 1
        return labeledFeatureVector


    def __str__(self):
        out = f""
        for event in self._events:
            out = out + f"\t{str(event)}\n"
        return out
    
    def __repr__(self):
        out = f""
        for event in self._events:
            out = out + f"\t{str(event)}\n"
        return out
