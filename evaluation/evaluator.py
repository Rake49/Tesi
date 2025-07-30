class Evaluator:
    def __init__(self, dataset, classifier):
        self._positiveFeatureTarget = "deviant"
        self._negativeFeatureTarget = "regular"
        self._truePositive = 0
        self._trueNegative = 0
        self._falsePositive = 0
        self._falseNegative = 0
        x, y = classifier.separateInputFromOutput(dataset)
        predictions = classifier.predict(x)
        assert len(y) == len(predictions)
        for i in range(len(y)):
            actual = y[i]
            predict = predictions[i]
            if actual == self._positiveFeatureTarget:
                if predict == self._positiveFeatureTarget:
                    self._truePositive += 1
                else:
                    self._falseNegative += 1
            else:
                if predict == self._negativeFeatureTarget:
                    self._trueNegative += 1
                else:
                    self._falsePositive += 1            

    def confusionMatrix(self):
        return [
            [self._trueNegative, self._falsePositive],
            [self._falseNegative, self._truePositive]
        ]

    def precision(self):
        return self._truePositive / (self._falsePositive + self._truePositive)
    
    def recall(self):
        return self._truePositive / (self._falseNegative + self._truePositive)
    
    def f1(self):
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())
    
    def positiveFeatureTarget(self):
        return self._positiveFeatureTarget
    
    def negativeFeatureTarget(self):
        return self._negativeFeatureTarget
    
    
