import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
def dataToList(filename):
  data = []
  with open(filename) as f:
    for line in f:
      data.append(line.split(" "))
  return np.array(data,float)

def linReg(mat):
  xs = mat[:,0]
  ys = mat[:,1]
  xs1 = np.column_stack((xs, np.ones((mat.shape[0],1))))
  return np.dot(np.dot(inv(np.dot(np.transpose(xs1), xs1)), np.transpose(xs1)), ys)

data = dataToList("./DanWood.dt")
result = linReg(data)
print result
squaredSum = 0
k = 0
for i in data:
  err = i[0] * result[0] + result[1] - i[1]
  squaredSum += err**2
  k += 1
print "Mean squared error: {0}".format(squaredSum/k)

def func(x):
  return x * result[0] + result[1]

plt.plot(data[:,0], data[:,1], 'b.', data[:, 0], data[:,0]*result[0]+result[1], 'k')
plt.show()
