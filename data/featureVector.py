class FeatureVector:
    def __init__(self, dimensione):
        self._vector = [0] * dimensione

    def incrementValue(self, pos: int):
        self._vector[pos] = self._vector[pos] + 1

    def __repr__(self):
        out = f""
        for el in self._vector:
            out = out + f"{el}, "
        return out
    
    def __str__(self):
        out = f""
        for el in self._vector:
            out = out + f"{el}, "
        return out