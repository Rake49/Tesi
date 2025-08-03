
import dice_ml
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Counterfactual:
    def __init__(self, trainSet, classifier, minPermittedRange, maxPermittedRange):
        self._minPermittedRange = minPermittedRange
        self._maxPermittedRange = maxPermittedRange
        le = LabelEncoder()
        x, y = classifier.separateInputFromOutput(trainSet)
        yEncoded = le.fit_transform(y)
        xdf = classifier.toPandasDF(x)
        ys = classifier.toPandasSeries(yEncoded)
        dfForDice = pd.concat([xdf, ys], axis = 1)
        diceData = dice_ml.Data(
            dataframe = dfForDice,
            continuous_features = classifier.columnsName(),
            outcome_name = classifier.targetFeatureName()
        )
        model = dice_ml.Model(
            model = classifier.model(),
            backend = 'sklearn'
        )
        self._exp = dice_ml.Dice(diceData, model, method = 'random')

    def generateCounterfactual(self, featureVector, columnsName, pathOut):
        xdf = pd.DataFrame(featureVector, columns = columnsName)
        currentPermittedRange = {}
        i = 0
        for feature in columnsName:
                currentPermittedRange[feature] = [self._minPermittedRange[i], self._maxPermittedRange]
                i += 1
        cf = self._exp.generate_counterfactuals(
            xdf,
            total_CFs = 3,
            desired_class = 'opposite',
            features_to_vary = columnsName,
            permitted_range = currentPermittedRange
        )
        cfDataframe = cf.cf_examples_list[0].final_cfs_df
        cfDataframe['Type'] = 'Counterfactual'
        originalInstanceDf = cf.cf_examples_list[0].test_instance_df
        originalInstanceDf['Type'] = 'Original'
        dfToExport = pd.concat([originalInstanceDf, cfDataframe])
        dfToExport.to_excel(pathOut, index = False)