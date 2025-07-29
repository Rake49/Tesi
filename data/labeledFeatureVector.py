from .featureVector import FeatureVector

class LabeledFeatureVector(FeatureVector):
    def __init__(self, label, dimensione):
        self._label = label
        super().__init__(dimensione)

    def label(self):
        return self._label
    
    def __repr__(self):
        return f"{super().__repr__()} LABEL: {self._label}"
    
    def __str__(self):
        return f"{super().__str__()} LABEL: {self._label}"