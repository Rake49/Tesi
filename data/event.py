from datetime import datetime

class Event:
    def __init__(self, activity: str, timestamp: datetime, label: str):
        self._activity = activity
        self._timestamp = datetime.strptime(timestamp, "%d/%m%Y %H:%M")
        self._label = label
    
    def activity(self):
        return self._activity