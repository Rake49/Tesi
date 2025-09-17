import plotly.figure_factory as ff
import pandas as pd
import os

def plotConfusionMatrix(evaluator, classifierName):
    labels = evaluator.labels()
    x = [f"Predicted {i}" for i in labels]
    y = [f"Actual {i}" for i in labels]
    confusionMatrix = evaluator.confusionMatrix()
    img = ff.create_annotated_heatmap(
        confusionMatrix,
        x=x,
        y=y,
        colorscale="Purples"
    )
    img.update_layout(
        title = dict(
            text = f"Matrice di confusione per {classifierName}",
            x = 0.5,
            xanchor = "center"
        ),
        yaxis=dict(autorange="reversed")
    )
    img.write_image(f"statistiche/confusionMatrixFor{classifierName}.png")


def exportMetricsToExcel(evaluatorsDict, pathOUT="statistiche/performanceMetrics.xlsx"):

    classifierNames = list(evaluatorsDict.keys())
    precisionScores = []
    recallScores = []
    f1Scores = []
    macroF1Scores = []
    accuracyScores = []

    for classifierName in classifierNames:
        eval = evaluatorsDict[classifierName]
        precisionScores.append(eval.precision())
        recallScores.append(eval.recall())
        f1Scores.append(eval.f1())
        macroF1Scores.append(eval.macroF1())
        accuracyScores.append(eval.accuracy())
    data = {
        'Precision': precisionScores,
        'Recall': recallScores,
        'F1-Score': f1Scores,
        'Macro F1': macroF1Scores,
        'Accuracy': accuracyScores
    }
    dfMetrics = pd.DataFrame(data, index = classifierNames)

    outputDir = os.path.dirname(pathOUT)
    if outputDir and not os.path.exists(outputDir):
        os.makedirs(outputDir)
    try:
        dfMetrics.to_excel(pathOUT, index=True)
    except Exception as e:
        print(f"Errore durante l'esportazione delle metriche in Excel: {e}")