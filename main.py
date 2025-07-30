from data import Log
from classifier import RandomForestClassifier, LGBMClassifier
from evaluation import Evaluator
from plot import plotConfusionMatrix

log = Log("sepsis_cases_1.csv", "fileConfig.json")
trainSetLog, testSetLog = log.split(42, 0.34)
trainSet = trainSetLog.transformToLabeledFeatureVectorList()
testSet = testSetLog.transformToLabeledFeatureVectorList()
randomForest = RandomForestClassifier(42, trainSetLog.dominio())
randomForest.fit(trainSet)
lgbm = LGBMClassifier(42, trainSetLog.dominio())
lgbm.fit(trainSet)
# print(randomForest.predict(testSet[0].featureVector()))
# print(lgbm.predict(testSet[0].featureVector()))
rfEval = Evaluator(testSet, randomForest)
lgbmEval = Evaluator(testSet, lgbm)
print("precision: ", rfEval.precision(), lgbmEval.precision())
print("recall: ", rfEval.recall(), lgbmEval.recall())
print("f1: ", rfEval.f1(), lgbmEval.f1())
plotConfusionMatrix(rfEval, "RandomForest")
# print(str(testSet))
# print(log.transformToLabeledFeatureVectorList())