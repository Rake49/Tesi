from data import Log
from classifier import RandomForestClassifier, LGBMClassifier

log = Log("sepsis_cases_1.csv", "fileConfig.json")
trainSetLog, testSetLog = log.split(42, 0.34)
trainSet = trainSetLog.transformToLabeledFeatureVectorList()
testSet = testSetLog.transformToLabeledFeatureVectorList()
randomForest = RandomForestClassifier(42)
randomForest.fit(trainSet)
lgbm = LGBMClassifier(42)
lgbm.fit(trainSet)
print(randomForest.predict(testSet[0].featureVector()))
print(lgbm.predict(testSet[0].featureVector()))

# print(str(testSet))
# print(log.transformToLabeledFeatureVectorList())