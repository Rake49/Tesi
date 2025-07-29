from data import Log
from pprint import pprint

log = Log("sepsis_cases_1.csv", "fileConfig.json")
trainSet, testSet = log.split(42, 0.34)
print(str(testSet))
# print(log.transformToLabeledFeatureVectorList())