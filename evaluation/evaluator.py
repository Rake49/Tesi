from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class Evaluator:
    def __init__(self, dataset, classifier, labels):
        x, self._actual = dataset.separateInputFromOutput()
        self._predictions = classifier.decode(classifier.predict(dataset.toPandasDF(x)))
        self._labels = labels
                  

    def confusionMatrix(self):
        return confusion_matrix(self._actual, self._predictions, labels = self._labels)

    def precision(self):
        return precision_score(self._actual, self._predictions, average = 'weighted')
    
    def recall(self):
        return recall_score(self._actual, self._predictions, average = 'weighted')
    
    def f1(self):
        return f1_score(self._actual, self._predictions, average = 'weighted')
    
    def labels(self):
        return self._labels
    
    def predictions(self):
        return self._predictions
    
    
