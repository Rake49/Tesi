from data import Log
from pprint import pprint

log = Log("sepsis_cases_1.csv", "fileConfig.json")
# print(str(log))
print(log.transformToLabeledFeatureVectorList())