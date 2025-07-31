
import dice_ml
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Counterfactual:
    def __init__(self, trainSet, classifier):
        le = LabelEncoder()
        self._MAX_PERMITTED_RANGE = 50
        x, y = classifier.separateInputFromOutput(trainSet)
        yEncoded = le.fit_transform(y)
        xdf = pd.DataFrame(x, columns = classifier.columnsName())
        ys = pd.Series(yEncoded, name = "Label")
        dfForDice = pd.concat([xdf, ys], axis = 1)
        diceData = dice_ml.Data(
            dataframe = dfForDice,
            continuous_features = classifier.columnsName(),
            outcome_name = 'Label'
        )
        model = dice_ml.Model(
            model = classifier.model(),
            backend = 'sklearn'
        )
        self._exp = dice_ml.Dice(diceData, model, method = 'random')

    def generateCounterfactual(self, testSet, classifier, pathOut, numExamplesToOutput = None):
        x, y = classifier.separateInputFromOutput(testSet)
        xdf = pd.DataFrame(x, columns = classifier.columnsName())
        if numExamplesToOutput is not None:
            xdf = xdf.head(numExamplesToOutput)
        allCfsDataframe = []
        for i in range(len(xdf)):
            queryInstance = xdf.iloc[i : i + 1] # Estrae una singola riga come DataFrame (formato richiesto da DiCE)
            currentPermittedRange = {}
            for feature in classifier.columnsName():
                currentValue = queryInstance[feature].iloc[0]
                currentPermittedRange[feature] = [currentValue, self._MAX_PERMITTED_RANGE]
            cf = self._exp.generate_counterfactuals(
                queryInstance,
                total_CFs = 2,
                desired_class = 'opposite',
                features_to_vary = classifier.columnsName(),
                permitted_range = currentPermittedRange
            )
            cfDataframe = cf.cf_examples_list[0].final_cfs_df
            cfDataframe['Type'] = 'Counterfactual'
            cfDataframe['Query_ID'] = i
            originalInstanceDf = cf.cf_examples_list[0].test_instance_df
            originalInstanceDf['Type'] = 'Original'
            originalInstanceDf['Query_ID'] = i
            combinedDf = pd.concat([originalInstanceDf, cfDataframe])
            allCfsDataframe.append(combinedDf)
        
        dfToExport = pd.concat(allCfsDataframe, ignore_index = True)
        dfToExport.to_excel(pathOut, index = False)