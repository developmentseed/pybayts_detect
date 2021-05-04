"""
Test to demonstrate the use of numpy to calculate mean and standarddeviation from a 3D masked array (simulation of a time series data cube).
Author: @developmentseed
"""

import numpy as np
import math

def variance(arr):
    n = len(arr)
    mean = sum(arr) / n
    deviations = [(x - mean) ** 2 for x in arr]
    variance = sum(deviations) / n
    return variance

def stdev(arr):
    var = variance(arr)
    std_dev = math.sqrt(var)
    return std_dev

def test_mean_std():
    arr1 = np.fromiter(range(16), dtype="int")
    arr2 =  np.reshape(arr1, (-1, 4))
    arr3 = np.pad(arr2, 2, mode='constant')
    arr4 = np.dstack([arr3, arr3, arr3])
    arr_mask_nodata = np.ma.masked_array(arr4, arr4 == 0)
    mean = np.mean(arr_mask_nodata) 
    std = np.std(arr_mask_nodata)  
    truemean = sum(arr1) / float(len(arr1))
    truesrd = stdev(arr1)
    assert mean == truemean
    assert std == truestd
