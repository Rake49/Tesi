
import dice_ml
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Counterfactual:
    def __init__(self, trainSet, classifier, minPermittedRange, maxPermittedRange):
        self._minPermittedRange = minPermittedRange
        self._maxPermittedRange = maxPermittedRange
        self._le = classifier.labelEncoder()
        x, y = trainSet.separateInputFromOutput()

        xdf = trainSet.toPandasDF(x)
        ys = trainSet.toPandasSeries(classifier._le.transform(y))

        dfForDice = pd.concat([xdf, ys], axis = 1)
        classMapping = {label: idx for idx, label in enumerate(classifier.model().classes_)}
        diceData = dice_ml.Data(
            dataframe = dfForDice,
            continuous_features = trainSet.columnsName(),
            outcome_name = trainSet.targetFeatureName(),
            class_mapping = classMapping
        )
        model = dice_ml.Model(
            model = classifier.model(),
            backend = 'sklearn'
        )
        self._exp = dice_ml.Dice(diceData, model, method = 'random')

    def generateCounterfactual(self, featureVector, columnsName, changeToOpposite, changeToLabel = None):
        xdf = pd.DataFrame(featureVector, columns = columnsName)
        currentPermittedRange = {}
        i = 0
        for feature in columnsName:
                currentPermittedRange[feature] = [self._minPermittedRange[i], self._maxPermittedRange[i]]
                i += 1

        cf = self._exp.generate_counterfactuals(
            xdf,
            total_CFs = 3,
            desired_class = ('opposite' if changeToOpposite else int(self._le.transform([changeToLabel])[0])),
            features_to_vary = columnsName,
            permitted_range = currentPermittedRange
        )
        print(type(cf))
        cfDataframe = cf.cf_examples_list[0].final_cfs_df
        cfDataframe['Label'] = self._le.inverse_transform(cfDataframe['Label'].astype(int))
        cfDataframe['Type'] = 'Counterfactual'
        originalInstanceDf = cf.cf_examples_list[0].test_instance_df
        originalInstanceDf['Type'] = 'Original'
        originalInstanceDf['Label'] = self._le.inverse_transform(originalInstanceDf['Label'].astype(int))
        return pd.concat([originalInstanceDf, cfDataframe])
    
    def exportToExcel(self, dfToExport, pathOut):
        dfToExport.to_excel(pathOut, index = False)