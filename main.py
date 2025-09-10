from data import Log
from classifier import RandomForestClassifier, LGBMClassifier
from evaluation import Evaluator
from plot import plotConfusionMatrix
from plot import exportMetricsToExcel
from counterfactual import Counterfactual

log = Log("sepsis_cases_1.csv", "fileConfig.json")
trainSetLog, testSetLog = log.split(0.66)
trainSet = trainSetLog.transformToLabeledFeatureVectorList()
testSet = testSetLog.transformToLabeledFeatureVectorList()

randomForest = RandomForestClassifier(42)
randomForest.fit(trainSet)

lgbm = LGBMClassifier(42)
lgbm.fit(trainSet)

rfEval = Evaluator(testSet, randomForest, log.labels())
lgbmEval = Evaluator(testSet, lgbm, log.labels())
print("precision: ", rfEval.precision(), lgbmEval.precision())
print("recall: ", rfEval.recall(), lgbmEval.recall())
print("f1: ", rfEval.f1(), lgbmEval.f1())

plotConfusionMatrix(rfEval, "RandomForest")
plotConfusionMatrix(lgbmEval, "LGBM")
exportMetricsToExcel({"Random Forest": rfEval, "LGBM": lgbmEval})

_, testSetLabels = testSet.separateInputFromOutput()
index = next((
    index for index, (pred, label) in enumerate(zip(rfEval.predictions(), testSetLabels)) 
    if pred == label and label == 'deviant'), None)
vectorForRF = testSet.dataset()[index]
index = next((
    index for index, (pred, label) in enumerate(zip(lgbmEval.predictions(), testSetLabels)) 
    if pred == label and label == 'deviant'), None)
vectorForLGBM = testSet.dataset()[index]

minPermittedRange = []
maxPermittedRange = []
for val in vectorForRF.featureVector():
    minPermittedRange.append(val)
    maxPermittedRange.append(50)

counterfactualForRF = Counterfactual(trainSet, randomForest, minPermittedRange, maxPermittedRange)
counterfactualDataFrameRF = counterfactualForRF.generateCounterfactual([vectorForRF.featureVector()], trainSet.columnsName(), True)
counterfactualForRF.exportToExcel(counterfactualDataFrameRF, "statistiche/counterfactualsRandomForest.xlsx")

counterfactualForLGBM = Counterfactual(trainSet, lgbm, minPermittedRange, maxPermittedRange)
counterfactualDataFrameLGBM = counterfactualForLGBM.generateCounterfactual([vectorForLGBM.featureVector()], trainSet.columnsName(), False, 'regular')
counterfactualForLGBM.exportToExcel(counterfactualDataFrameLGBM, "statistiche/counterfactualsLGBM.xlsx")