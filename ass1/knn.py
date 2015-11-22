import numpy as np
import sys

def dataToList(filename):
  retArray = []
  retClass = []
  with open(filename) as f:
    for line in f:
      k = line.split(" ")
      retArray.append(k[0])
      retArray.append(k[1])
      retClass.append(k[2])
  return (np.array(retArray,float).reshape((len(retClass),2)),np.array(retClass,int))

def knn(k, trainSetData, trainSetClass, target, distFun):
  dist = distFun(trainSetData, target)
  res = np.argsort(dist, axis=0)[0:k]
  counts = np.bincount(trainSetClass[res])
  return np.argmax(counts)

def euclideanNoSqrt(a,b):
  dist = (a - b) ** 2
  return np.sqrt(np.sum(dist, axis = 1))

trainData = dataToList("./IrisTrainML.dt")
testData = dataToList("./IrisTestML.dt")


if sys.argv[1] == "1.1":
  for k in [1,3,5]:
    err = 0.0
    for i in range(0,38):
      if knn(k, trainData[0], trainData[1], testData[0][i], euclideanNoSqrt) != testData[1][i]:
        err += 1.0
    print "Error of the {0}-nn classifier: {1}".format(k, + err / 38)

def nFoldCrossValidation(n, k, trainSetData, trainSetClass):
  dataSplit = np.split(trainSetData,n)
  classSplit = np.split(trainSetClass,n)
  errTotal = 0.0
  for i in range(0,n):
    trData = np.concatenate(dataSplit[:i] + dataSplit[(i + 1):])
    trClass = np.concatenate(classSplit[:i] + classSplit[(i + 1):])
    teData = dataSplit[i]
    teClass = classSplit[i]
    err = 0.0
    for j in range(0,len(teClass)):
      if knn(k, trData, trClass, teData[j], euclideanNoSqrt) != teClass[j]:
        err += 1.0
    errTotal += err / len(teClass)
  return errTotal/n

if sys.argv[1] == "1.2":
  print "Crossvalidation error for the k-nn classifier for k=[1,2 ... 25]"
  for k in range(1,26):
      print "{0}-nn: {1}".format(k, nFoldCrossValidation(5, k, trainData[0], trainData[1]))


