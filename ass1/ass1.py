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
    print "Error of the {0}-nn classifier: {1}".format(k, err / 38)

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
  print "Crossvalidation error for the k-nn classifier for k=[1,3 ... 25]"
  bestk = 0
  besterr = np.inf
  for k in [x for x in range(1,26) if x % 2]:
      err = nFoldCrossValidation(5, k, trainData[0], trainData[1])
      if err < besterr:
        besterr = err
        bestk = k
      print "{0}-nn: {1}".format(k, err)
  print "\nBest k found was {0} with an error of {1}".format(bestk,besterr)

def normFunc(data):
  stdDev = np.sqrt(np.var(data))
  mean = np.mean(data)
  def returnFunc(x):
    return (x - mean) / stdDev
  return returnFunc

if sys.argv[1] == "1.3":
  f1 = trainData[0][:,0]
  f2 = trainData[0][:,1]
  print "Original training data:"
  print "First feature, mean: {0}, variance: {1}".format(np.mean(f1),np.var(f1))
  print "Second feature, mean: {0}, variance: {1}".format(np.mean(f2),np.var(f2))
  f1norm = normFunc(f1)
  f2norm = normFunc(f2)
  f1testNorm = f1norm(testData[0][:,0])
  f2testNorm = f2norm(testData[0][:,1])
  f1trainNorm = f1norm(f1)
  f2trainNorm = f2norm(f2)
  print "\nNormalized test data:"
  print "First feature, mean: {0}, variance: {1}".format(np.mean(f1testNorm),np.var(f1testNorm))
  print "Second feature, mean: {0}, variance: {1}".format(np.mean(f2testNorm),np.var(f2testNorm))
  err = 0.0
  testDataNorm = np.column_stack((f1testNorm,f2testNorm))
  trainDataNorm = np.column_stack((f1trainNorm,f2trainNorm))
  print "\nCrossvalidation error for the k-nn classifier for k=[1,3 ... 25] on normalized test data"
  bestk = 0
  besterr = np.inf
  for k in [x for x in range(1,26) if x % 2]:
      err = nFoldCrossValidation(5, k, trainDataNorm, trainData[1])
      if err <= besterr:
        besterr = err
        bestk = k
      print "{0}-nn: {1}".format(k, err)
  print "\nBest k found was {0} with an error of {1}".format(bestk,besterr)
  for i in range(0,38):
    if knn(bestk, trainDataNorm, trainData[1], testDataNorm[i], euclideanNoSqrt) != testData[1][i]:
      err += 1.0
  print "Error of the {0}-nn classifier on normalized data: {1}".format(bestk, err / 38)

