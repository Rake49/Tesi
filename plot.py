import plotly.figure_factory as ff

def plotConfusionMatrix(evaluator, classifierName):
    labels = [evaluator.negativeFeatureTarget(), evaluator.positiveFeatureTarget()]
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