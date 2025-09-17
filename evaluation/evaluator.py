from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

class Evaluator:
    def __init__(self, dataset, classifier, labels):
        x, self._actual = dataset.separateInputFromOutput()
        self._predictions = classifier.decode(classifier.predict(dataset.toPandasDF(x)))
        self._labels = labels
        # weights = classifier.weights()
        # self._sampleWeights = np.array([weights[label] for label in self._actual])
        self._sampleWeights = compute_sample_weight(class_weight='balanced', y=self._actual)
                  

    def confusionMatrix(self):
        return confusion_matrix(self._actual, self._predictions, labels = self._labels)

    def precision(self):
        return precision_score(self._actual, self._predictions, average = 'weighted')
    
    def recall(self):
        return recall_score(self._actual, self._predictions, average = 'weighted')
    
    def f1(self):
        return f1_score(self._actual, self._predictions, average = 'weighted')
    
    def macroF1(self):
        return f1_score(self._actual, self._predictions, average = 'macro')
    
    def accuracy(self):
        return accuracy_score(self._actual, self._predictions, sample_weight=self._sampleWeights)
    
    def labels(self):
        return self._labels
    
    def predictions(self):
        return self._predictions
    
    
