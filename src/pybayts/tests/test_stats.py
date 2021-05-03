"""
Test to demonstrate the use of numpy to calculate mean and standarddeviation from a 3D masked array (simulation of a time series data cube).
Author: @developmentseed
"""

import numpy as np

arr1 = np.random.random((100,100))
arr2 = np.pad(arr1, 20, mode='constant')
arr3 = np.dstack([arr2, arr2, arr2])
arr_mask_nodata = np.ma.masked_array(arr3, arr3 == 0)
mean = np.mean(arr_mask_nodata) 
std = np.std(arr_mask_nodata)  
print(mean,std)
