import pickle
import datetime
import random

import matplotlib.pyplot as plt
import csv
from entity.EdgeDevice import *

from function.Initialization import legalPathsInit

currtime = datetime.datetime.now()
time_str = datetime.datetime.strftime(currtime, '%Y-%m-%d-%H-%M-%S')

file = 'data/Data2022-05-06-11-23-36.pkl'
f = open(file, 'rb')
dataList1 = pickle.load(f)
f.close()
serviceList = dataList1[0]
deviceList = dataList1[1]
capacity = dataList1[2]
Z = dataList1[3]
allLegalPaths = dataList1[4]

Z[9][6] = 0
Z[6][9] = Z[9][6]
Z[3][9] = 5.3
Z[9][3] = Z[4][9]
Z[4][8] = 5.3
Z[8][4] = Z[4][8]



allLegalPaths = legalPathsInit(Z)


dataList = []
dataList.append(serviceList)
dataList.append(deviceList)
dataList.append(capacity)
dataList.append(Z)
dataList.append(allLegalPaths)
str = 'data/'+ 'Data' + time_str + '.pkl'
f = open(str, 'wb+')
pickle.dump(dataList, f)
f.close()