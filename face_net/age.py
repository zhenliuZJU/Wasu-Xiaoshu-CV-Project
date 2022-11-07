# coding:UTF-8
'''
Created on 2015年5月12日
@author: zhaozhiyong
'''

import scipy.io as scio
import h5py
import numpy as np
dataFile = '../../../data/CACD/data/celebrity2000.mat'
data = h5py.File(dataFile)
# data = scio.loadmat(dataFile)
for key in data.keys():
    a = np.array(data[key])
    print(f"{key}||{a}")
print(type(data))