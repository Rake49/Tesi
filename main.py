from data import Log
from classifier import RandomForestClassifier, LGBMClassifier
from evaluation import Evaluator
from plot import plotConfusionMatrix
from plot import exportMetricsToExcel
from counterfactual import Counterfactual

log = Log("sepsis_cases_1.csv", "fileConfig.json")
trainSetLog, testSetLog = log.split(42, 0.34)
trainSet = trainSetLog.transformToLabeledFeatureVectorList()
testSet = testSetLog.transformToLabeledFeatureVectorList()

randomForest = RandomForestClassifier(42, trainSetLog.dominio())
randomForest.fit(trainSet)

lgbm = LGBMClassifier(42, trainSetLog.dominio())
lgbm.fit(trainSet)

rfEval = Evaluator(testSet, randomForest)
lgbmEval = Evaluator(testSet, lgbm)
print("precision: ", rfEval.precision(), lgbmEval.precision())
print("recall: ", rfEval.recall(), lgbmEval.recall())
print("f1: ", rfEval.f1(), lgbmEval.f1())

plotConfusionMatrix(rfEval, "RandomForest")
exportMetricsToExcel({"Random Forest": rfEval, "LGBM": lgbmEval})

counterfactualForRF = Counterfactual(trainSet, randomForest)
counterfactualForRF.generateCounterfactual(testSet, randomForest, 20)