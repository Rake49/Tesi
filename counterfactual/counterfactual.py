
import dice_ml
import pandas as pd
import numpy as np

class Counterfactual:
    def __init__(self, trainSet, classifier):
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

    def generateCounterfactual(self, featureVector, caseID, columnsName, minPermittedRange, maxPermittedRange, changeToOpposite, changeToLabel = None):
        xdf = pd.DataFrame(featureVector, columns = columnsName)
        currentPermittedRange = {}
        i = 0
        for feature in columnsName:
                currentPermittedRange[feature] = [minPermittedRange[i], maxPermittedRange[i]]
                i += 1

        cf = self._exp.generate_counterfactuals(
            xdf,
            total_CFs = 10,
            desired_class = ('opposite' if changeToOpposite else int(self._le.transform([changeToLabel])[0])),
            features_to_vary = columnsName,
            permitted_range = currentPermittedRange
        )
        cfDataframe = self.closestCounterfactual(featureVector, columnsName, cf.cf_examples_list[0].final_cfs_df)
        cfDataframe['Predicted'] = self._le.inverse_transform(cfDataframe['Label'].astype(int))
        cfDataframe['Type'] = 'Counterfactual'
        cfDataframe['CaseID'] = ""
        cfDataframe['PrefixLength'] = ""
        originalInstanceDf = cf.cf_examples_list[0].test_instance_df
        originalInstanceDf['Type'] = 'Original'
        originalInstanceDf['Predicted'] = self._le.inverse_transform(originalInstanceDf['Label'].astype(int))
        originalInstanceDf['CaseID'] = caseID
        originalInstanceDf['PrefixLength'] = sum(featureVector[0])
        return pd.concat([originalInstanceDf, cfDataframe])
    
    def closestCounterfactual(self, originalVector, columnsNames, cf):
        originalVectorArray = np.array(originalVector, dtype=float)
        cfArrays = cf[columnsNames].to_numpy(dtype=float)
        distances = [np.linalg.norm(originalVectorArray - cfArray) for cfArray in cfArrays]
        bestIndex = np.argmin(distances)
        return cf.iloc[[bestIndex]].copy()
         