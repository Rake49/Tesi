from data import Log
from classifier import RandomForestClassifier, LGBMClassifier
from evaluation import Evaluator
from plot import plotConfusionMatrix
from plot import exportMetricsToExcel
from counterfactual import Counterfactual

log = Log("sepsis_cases_1.csv", "fileConfig.json")
trainSetLog, testSetLog = log.split(42, 0.66)
trainSet = trainSetLog.transformToLabeledFeatureVectorList()
testSet = testSetLog.transformToLabeledFeatureVectorList()

randomForest = RandomForestClassifier(42, trainSetLog.dominio(), 'Label')
randomForest.fit(trainSet)

lgbm = LGBMClassifier(42, trainSetLog.dominio(), 'Label')
lgbm.fit(trainSet)

rfEval = Evaluator(testSet, randomForest, log.labels())
lgbmEval = Evaluator(testSet, lgbm, log.labels())
print("precision: ", rfEval.precision(), lgbmEval.precision())
print("recall: ", rfEval.recall(), lgbmEval.recall())
print("f1: ", rfEval.f1(), lgbmEval.f1())

plotConfusionMatrix(rfEval, "RandomForest")
exportMetricsToExcel({"Random Forest": rfEval, "LGBM": lgbmEval})

minPermittedRange = []
for val in testSet[0].featureVector():
    minPermittedRange.append(val)

counterfactualForRF = Counterfactual(trainSet, randomForest, minPermittedRange, 50)
counterfactualForRF.generateCounterfactual([testSet[0].featureVector()], randomForest.columnsName(), "statistiche/counterfactualsRandomForest.xlsx")

counterfactualForLGBM = Counterfactual(trainSet, lgbm, minPermittedRange, 50)
counterfactualForLGBM.generateCounterfactual([testSet[0].featureVector()], lgbm.columnsName(), "statistiche/counterfactualsLGBM.xlsx")