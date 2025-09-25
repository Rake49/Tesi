from data import Log
from classifier import RandomForestClassifier, XGBoostClassifier
from evaluation import Evaluator
from plot import plotConfusionMatrix
from plot import exportMetricsToExcel
from plot import exportClassificationReportToExcel
from counterfactual import Counterfactual
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def getCFTraceIDForClassifier(dataset, classifier, wantedPrediction, originalLabel):
    caseIDList = dataset.caseIDDominio()
    for caseID in caseIDList:
        lfv = (dataset.selectCaseID(caseID))[-1][1]
        label = lfv.label()
        fv = lfv.featureVector()
        pred = classifier.decode(classifier.predict(dataset.toPandasDF([fv])))[0]
        if pred == wantedPrediction and label == originalLabel:
            return caseID

def minPermittedRange(featureVector):
    return featureVector

def maxPermittedRange(featureVector):
    max = 0
    for val in featureVector:
        if val > max:
            max = val
    return [max + 5] * len(featureVector)

def alternate_rows_style(row):
    return ['background-color: #faf5e9' if row.name % 2 != 1 else '' for _ in row]

def highlight_deviant(val):
    if isinstance(val, str) and 'deviant' in val.lower():
        return 'background-color: #ffcccb'
    return ''

def main():
    log = Log("sepsis_cases_1.csv", "fileConfig.json")
    trainSetLog, testSetLog = log.split(0.66)
    trainSet = trainSetLog.transformToLabeledFeatureVectorList()
    testSet = testSetLog.transformToLabeledFeatureVectorList()

    randomForest = RandomForestClassifier(42, {'deviant': 9, 'regular': 1})
    randomForest.fit(trainSet)

    xgb = XGBoostClassifier(42, {'deviant': 9, 'regular': 1})
    xgb.fit(trainSet)

    rfEval = Evaluator(testSet, randomForest, log.labels())
    xgbEval = Evaluator(testSet, xgb, log.labels())
    print("precision: ", rfEval.precision(), xgbEval.precision())
    print("recall: ", rfEval.recall(), xgbEval.recall())
    print("f1: ", rfEval.f1(), xgbEval.f1())

    plotConfusionMatrix(rfEval, "RandomForest")
    plotConfusionMatrix(xgbEval, "XGBoost")
    exportMetricsToExcel({"Random Forest": rfEval, "XGBoost": xgbEval})
    exportClassificationReportToExcel({"Random Forest": rfEval, "XGBoost": xgbEval})

    classifiers = [randomForest, xgb]
    labels = ['deviant', 'regular']
    for classifier in classifiers:
        for label in labels:
            ID = getCFTraceIDForClassifier(testSet, classifier, 'deviant', label)
            if ID == None:
                continue
            subtraces = testSet.selectCaseID(ID)
            listForDataFrame = []
            for caseID, labeledFeatureVector in subtraces:
                counterfactual = Counterfactual(trainSet, classifier)
                cfDataFrame = counterfactual.generateCounterfactual(
                    [labeledFeatureVector.featureVector()], 
                    caseID, 
                    trainSet.columnsName(),
                    minPermittedRange(labeledFeatureVector.featureVector()), 
                    maxPermittedRange(labeledFeatureVector.featureVector()),  
                    False,
                    'regular')
                cfDataFrame['Actual'] = label
                allColumns = list(cfDataFrame.columns)
                allColumns.remove('CaseID')
                allColumns.remove('PrefixLength')
                allColumns.remove('Actual')
                allColumns.remove('Predicted')
                allColumns.remove('Type')
                allColumns.remove('Label')
                newOrder = ['CaseID', 'PrefixLength'] + allColumns + ['Actual', 'Predicted', 'Type']
                cfDataFrame = cfDataFrame[newOrder]
                listForDataFrame.append(cfDataFrame)
            finalDataFrame = pd.concat(listForDataFrame, ignore_index=True)
            pathOut = f"statistiche/counterfactuals_{classifier.name()}_actual{label}_predDeviant.xlsx"
            styler = finalDataFrame.style
            styler = styler.apply(alternate_rows_style, axis=1)
            styler = styler.map(highlight_deviant, subset=['Actual', 'Predicted'])
            styler.to_excel(pathOut, engine='openpyxl', index = False)

if __name__ == "__main__":
    main()

    