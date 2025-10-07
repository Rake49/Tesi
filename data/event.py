from datetime import datetime

class Event:
    def __init__(self, activity: str, timestamp: str):
        self._activity = activity
        self._timestamp = datetime.strptime(timestamp, "%d/%m/%Y %H:%M")
    
    def activity(self):
        return self._activity
    
    def timestamp(self):
        return self._timestamp
    
    def __str__(self):
        return f"ATTIVITA: '{self._activity}'; TIMESTAMP: '{self._timestamp}'"