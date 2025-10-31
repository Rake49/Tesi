from data import Log
from classifier import RandomForestClassifier, XGBoostClassifier
from evaluation import Evaluator
from plot import plotConfusionMatrix
from plot import exportMetricsToExcel
from plot import exportClassificationReportToExcel
from counterfactual import Counterfactual
import pandas as pd
import os
import shutil
import random
import numpy as np
from statistics import mean

def set_seed(seed):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)  eseguire $env:PYTHONHASHSEED = "42" nel terminale prima dell'esecuzione
    np.random.seed(seed)

set_seed(42)

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

def maxPermittedRange(featureVector, max):
    return [value + max for value in featureVector]

def alternate_rows_style(row):
    # return ['background-color: #faf5e9' if row.name % 2 != 1 else '' for _ in row]
    # e9ecef
    if row['Type'] == 'Original':
        return ['background-color: #e9ecef'] * len(row)
    else:
        return [''] * len(row)

def highlight_deviant(val):
    # fff3cd
    # c3e6cb
    if isinstance(val, str) and 'deviant' in val.lower():
        return 'background-color: #fff3cd'
    return ''

def highlight_unable(val):
    # ffe0b2
    if isinstance(val, str) and 'unable' in val.lower():
        return 'background-color: #ffcccb'
    return ''

def dataFrameForPrefix(caseID, pred, fv, columnsNames, type):
    cfDataFrame = pd.DataFrame([fv], columns=columnsNames)
    cfDataFrame['CaseID'] = caseID
    cfDataFrame['PrefixLength'] = sum(fv)
    cfDataFrame['Predicted'] = pred
    cfDataFrame['Type'] = type
    cfDataFrame['Label'] = ""
    return cfDataFrame


def main(datasetName, fileConfig, rfWeights, xgbWeights, max):
    log = Log(datasetName, fileConfig)
    trainSetLog, testSetLog = log.split(0.66)
    trainSet = trainSetLog.transformToLabeledFeatureVectorList()
    testSet = testSetLog.transformToLabeledFeatureVectorList()
    randomForest = RandomForestClassifier(42, rfWeights)
    randomForest.fit(trainSet)

    xgb = XGBoostClassifier(42, xgbWeights)
    xgb.fit(trainSet)

    rfEval = Evaluator(testSet, randomForest, log.labels())
    xgbEval = Evaluator(testSet, xgb, log.labels())
    print("precision: ", rfEval.precision(), xgbEval.precision())
    print("recall: ", rfEval.recall(), xgbEval.recall())
    print("f1: ", rfEval.f1(), xgbEval.f1())

    plotConfusionMatrix(rfEval, f"RandomForest-{datasetName}")
    plotConfusionMatrix(xgbEval, f"XGBoost-{datasetName}")
    # exportMetricsToExcel({"Random Forest": rfEval, "XGBoost": xgbEval})
    # exportClassificationReportToExcel({"Random Forest": rfEval, "XGBoost": xgbEval})

    info = {}

    # xtn, xfp, xfn, xtp = xgbEval.confusionMatrix().ravel().tolist()
    # xClassRep = xgbEval.classReport()
    # xpd = xClassRep['deviant']['precision']
    # xpr = xClassRep['regular']['precision']
    # xpmacro = xClassRep['macro avg']['precision']
    # xpweigh = xClassRep['weighted avg']['precision']
    # xrd = xClassRep['deviant']['recall']
    # xrr = xClassRep['regular']['recall']
    # xrmacro = xClassRep['macro avg']['recall']
    # xrweigh = xClassRep['weighted avg']['recall']
    # xf1d = xgbEval.f1Deviant()
    # xf1r = xClassRep['regular']['f1-score']
    # xf1macro = xClassRep['macro avg']['f1-score']
    # xf1weigh = xClassRep['weighted avg']['f1-score']
    # xaccuracy = xClassRep['accuracy']
    # info['x'] = [xtp, xtn, xfp, xfn, xpd, xpr, xpmacro, xpweigh, xrd, xrr, xrmacro, xrweigh, xf1d, xf1r, xf1macro, xf1weigh, xaccuracy]

    # rtn, rfp, rfn, rtp = rfEval.confusionMatrix().ravel().tolist()
    # rClassRep = rfEval.classReport()
    # rpd = rClassRep['deviant']['precision']
    # rpr = rClassRep['regular']['precision']
    # rpmacro = rClassRep['macro avg']['precision']
    # rpweigh = rClassRep['weighted avg']['precision']
    # rrd = rClassRep['deviant']['recall']
    # rrr = rClassRep['regular']['recall']
    # rrmacro = rClassRep['macro avg']['recall']
    # rrweigh = rClassRep['weighted avg']['recall']
    # rf1d = rfEval.f1Deviant()
    # rf1r = rClassRep['regular']['f1-score']
    # rf1macro = rClassRep['macro avg']['f1-score']
    # rf1weigh = rClassRep['weighted avg']['f1-score']
    # raccuracy = rClassRep['accuracy']
    # info['r'] = [rtp, rtn, rfp, rfn, rpd, rpr, rpmacro, rpweigh, rrd, rrr, rrmacro, rrweigh, rf1d, rf1r, rf1macro, rf1weigh, raccuracy]

    # classifiers = [randomForest, xgb]
    # for classifier in classifiers:
    #     path = f"statistiche/bpic11{classifier.name()}CounterfactualsWithMax_{max}"
    #     if os.path.exists(path):
    #         shutil.rmtree(path)
    #     caseIDList = testSet.caseIDDominio()
    #     numPredDev = 0
    #     numTruePos = 0
    #     numFalsePos = 0
    #     numCFOfTrue = 0
    #     numCFOfFalse = 0
    #     distancesForTrue = []
    #     i = 0
    #     counterfactual = Counterfactual(trainSet, classifier)
    #     for caseID in caseIDList:
    #         i += 1
    #         print(i, caseID)
    #         subtraces = testSet.selectCaseID(caseID)
    #         listForDataFrame = []
    #         allRegular = True
    #         someUnable = False
    #         for _, labeledFeatureVector in subtraces:
    #             isTruePos = False
    #             fv = labeledFeatureVector.featureVector()
    #             label = labeledFeatureVector.label()
    #             pred = classifier.decode(classifier.predict(testSet.toPandasDF([fv])))[0]
    #             if pred == 'regular':
    #                 cfDataFrame = dataFrameForPrefix(caseID, pred, fv, testSet.columnsName(), 'Original')
    #             else:
    #                 allRegular = False
    #                 numPredDev += 1
    #                 if label == 'deviant':
    #                     numTruePos += 1
    #                     isTruePos = True
    #                 else:
    #                     numFalsePos += 1
    #                 cfDataFrame, distance = counterfactual.generateCounterfactual(
    #                     [labeledFeatureVector.featureVector()], 
    #                     caseID, 
    #                     trainSet.columnsName(),
    #                     minPermittedRange(labeledFeatureVector.featureVector()), 
    #                     maxPermittedRange(labeledFeatureVector.featureVector(), max),  
    #                     False,
    #                     'regular')
    #                 if cfDataFrame is False:
    #                     someUnable = True
    #                     cfDataFrame = pd.concat([dataFrameForPrefix(caseID, pred, fv, testSet.columnsName(), 'Original'), 
    #                                             dataFrameForPrefix(caseID, 'Unable', [np.nan]*len(testSet.columnsName()), testSet.columnsName(), 'Counterfactual')], 
    #                                             ignore_index=True)
    #                 else:
    #                     if isTruePos:
    #                         numCFOfTrue += 1
    #                         distancesForTrue.append(distance)
    #                     else:
    #                         numCFOfFalse += 1
    #             cfDataFrame['Actual'] = label
    #             allColumns = list(cfDataFrame.columns)
    #             allColumns.remove('CaseID')
    #             allColumns.remove('PrefixLength')
    #             allColumns.remove('Actual')
    #             allColumns.remove('Predicted')
    #             allColumns.remove('Type')
    #             allColumns.remove('Label')
    #             newOrder = ['CaseID', 'PrefixLength'] + allColumns + ['Actual', 'Predicted', 'Type']
    #             cfDataFrame = cfDataFrame[newOrder]
    #             listForDataFrame.append(cfDataFrame)
    #         finalDataFrame = pd.concat(listForDataFrame, ignore_index=True)

    #         pathOut = path + f"/{'#' if someUnable else ''}{'_' if allRegular else ''}{caseID}.xlsx"
    #         outputDir = os.path.dirname(pathOut)
    #         if outputDir and not os.path.exists(outputDir):
    #             os.makedirs(outputDir)
    #         styler = finalDataFrame.style
    #         styler = styler.apply(alternate_rows_style, axis=1)
    #         styler = styler.map(highlight_deviant, subset=['Actual', 'Predicted'])
    #         styler = styler.map(highlight_unable, subset=['Predicted'])
    #         styler.to_excel(pathOut, engine='openpyxl', index = False)
    #     info[classifier.name()] = [numPredDev, numTruePos, numFalsePos, numCFOfTrue, numCFOfFalse, mean(distancesForTrue)]
    return info

if __name__ == "__main__":
    # "sepsis_cases_1.csv", "SEPSISfileConfig.json", {'deviant': 9, 'regular': 1}, {'deviant': 9, 'regular': 1}
    # "bpic2012_O_ACCEPTED-COMPLETE.csv", "BPIC2012fileConfig.json", {'deviant': 2, 'regular': 1}, {'deviant': 2, 'regular': 1}
    # "BPIC11_f1.csv", "BPIC11fileConfig.json", {'deviant': 7, 'regular': 1}, {'deviant': 2, 'regular': 1}

    columns = ["Predetti deviant", "True Positive", "False Positive", "Numero di cf generati sui true positive", 
               "Numero di cf generati sui false positive", "Distanza media nei true positive"]
    dfList = []
    for max in range(1, 6):
        info = main("sepsis_cases_1.csv", "SEPSISfileConfig.json", {'deviant': 9, 'regular': 1}, {'deviant': 9, 'regular': 1}, max)
        df = pd.DataFrame.from_dict(info, orient='index', columns=columns)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Classificatore'}, inplace=True)
        df['Valore massimo per i counterfactual'] = max
        dfList.append(df)
    outPath = "statistiche/sepsissummaryReport.xlsx"
    finalDF = pd.concat(dfList, ignore_index=True)
    columnOrder = ['Valore massimo per i counterfactual', 'Classificatore'] + columns
    finalDF = finalDF[columnOrder]
    finalDF.to_excel(outPath, index=False)

    # columns = ['Classificatore', 'Peso deviant', 'Peso regular', 'TP', 'TN', 'FP', 'FN', 'Precision Deviant', 'Precision Regular', 'Precision Macro Avg', 'Precision Weighted Avg', 'Recall Deviant', 'Recall Regular', 'Recall Macro Avg', 'Recall Weighted Avg', 'F1 Deviant',	'F1 Regular', 'F1 Macro Avg', 'F1 Weighted Avg', 'Accuracy']
    

    # dfList = []
    # for i in range(1, 11):
    #     for j in range(1, 11):
    #         info = main("sepsis_cases_1.csv", "SEPSISfileConfig.json", {'deviant': i, 'regular': j}, {'deviant': i, 'regular': j}, 1)
    #         row1 = ['Random Forest', i, j] + info['r']
    #         row2 = ['XGBoost', i, j] + info['x']
    #         dfList.append(pd.DataFrame([row1], columns=columns))
    #         dfList.append(pd.DataFrame([row2], columns=columns))
    # finalDF = pd.concat(dfList, ignore_index=True)
    # finalDF.to_excel("statistiche/sepsisMetriche.xlsx", index=False)