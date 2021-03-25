"""
Module to evaluate Bayts algorithm against reference data in our study AOIs.
Author: @developmentseed
"""  

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import xarray as xr
import rioxarray as rx

def generate_cm(flat_groundtruth, flat_decimal_yr_arr, year):
    """Compute confusion matrix using co-registered reference and bayts inference data.
    Args:
        flat_groundtruth (array): flattened array of boolean values where [True = change, False = no change]
        for a given year
        
        flat_decimal_yr_arr (array): flattened array of boolean values where [True = change, False = no change]
        for a given year
        
        year (integer): the year associated with the flat_groundtruth and flat_decimal_yr_arr arrays.
        
    Returns:
        A confusion matrix for a single year.
    """
    
    labels = [False,  True]
    cm = confusion_matrix(flat_groundtruth, flat_decimal_yr_arr, labels=labels)
    print(f"Confusion matrix for {year}:", cm)
    return cm

def generate_f1(flat_groundtruth, flat_decimal_yr_arr, year):
    """Compute F1 score using co-registered reference and bayts inference data.
    Args:
        flat_groundtruth (array): flattened array of boolean values where [True = change, False = no change]
        for a given year
        
        flat_decimal_yr_arr (array): flattened array of boolean values where [True = change, False = no change]
        for a given year
        
        year (integer): the year associated with the flat_groundtruth and flat_decimal_yr_arr arrays.
        
    Returns:
        The F1 score for a single year.
    """
    
    f1 = f1_score(flat_groundtruth, flat_decimal_yr_arr, average='macro')
    print(f"F1 score for {year}:", f1)
    return f1

def evaluate(groundtruth, decimal_yr_arr, aoi_name):
    """Evaluate co-registered reference and bayts inference data using confusion matrix and F1 score.
    Args:
        groundtruth (str): rioxarray of clipped reference image (may need to run reproject match against a sample time series mosaic for the AOI)
        
        decimal_yr_arr (array): array of boolean values where [True = change, False = no change]
        
        aio_name (str): the area of interest to evaluate, which can be one of ['Brazil', 'DRC', 'Indonesia']. 
        
    Returns:
        Printed confusion matrices and F1 scores for each year in the study period.
    """
    
    decimal_yr_arr_years = np.unique(decimal_yr_arr.astype(np.uint16))
    
    if aoi_name == 'Brazil':
        # terrabrasilia ground truth years
        class_year_dict = {0:2008, 1:2009, 2:2010, 3:2011, 4:2012, 5:2013, 6:2014, 7:2015, 8:2016, 9:2017, 10:2018, 11:2019}

        
        year_gt_list = []
        year_pr_list = []
        
        for cl in class_year_dict:          
            year_gt = class_year_dict[cl]
            year_gt_list.append(year_gt)
            
        for yr in decimal_yr_arr_years.tolist():
            year_pr_list.append(yr)
        
        match_years = set(year_gt_list) & set(year_pr_list)
        
        print("year_gt_list, year_pr_list: ", year_gt_list, year_pr_list)
        print("match_years: ", match_years)
        
        for year in match_years:
            groundtruth_arr = groundtruth.copy()
            groundtruth_arr = groundtruth_arr  > cl
        
            groundtruth_flat = groundtruth_arr.values.flatten()
            decimal_yr_arr_flat = decimal_yr_arr.astype(np.uint16).flatten()
            
            cm = generate_cm(groundtruth_flat, decimal_yr_arr_flat, year)
            assert(cm.shape == (2, 2))
            f1 = generate_f1(groundtruth_flat, decimal_yr_arr_flat, year)

    else:
        # aoi is either DRC or Indonesia
        # Global Forest Watch ground truth years
        class_year_dict = {0:2000, 1:2001, 2:2002, 3:2003, 4:2004, 5:2005, 6:2006, 7:2007, 8:2008, 9:2009, 10:2010, 11:2011, 12:2012, 13:2013, 14:2014, 15:2015, 16:2016, 17:2017, 18:2018, 19:2019}

        
        year_gt_list = []
        year_pr_list = []
        
        for cl in class_year_dict:          
            year_gt = class_year_dict[cl]
            year_gt_list.append(year_gt)
            
        for yr in decimal_yr_arr_years.tolist():
            year_pr_list.append(yr)
        
        match_years = set(year_gt_list) & set(year_pr_list)
        print("match_years: ", match_years)
        
        for year in match_years:
            groundtruth_arr = groundtruth.copy()
            groundtruth_arr = groundtruth_arr  > cl
        
            groundtruth_flat = groundtruth_arr.values.flatten()
            decimal_yr_arr_flat = decimal_yr_arr.astype(np.uint16).flatten()
            decimal_yr_arr_flat = decimal_yr_arr_flat.astype(bool)
            
            print("unique values in groundtruth_flat, decimal_yr_arr_flat: ", np.unique(groundtruth_flat), np.unique(decimal_yr_arr_flat))
            
            cm = generate_cm(groundtruth_flat, decimal_yr_arr_flat, year)
            assert(cm.shape == (2, 2))
            f1 = generate_f1(groundtruth_flat, decimal_yr_arr_flat, year)

