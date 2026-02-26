#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Metadata:
#   Date    : 2025-02-18
#   Author  : [Your Name]
#   Description:
#       This script performs analysis on the recovered trigger mask norms for 
#       tabular data to detect potential backdoors. Recovered masks are stored 
#       as CSV files, with each file containing a 1D vector (one row of features).
#
#       The script computes the L1 norm of each mask and uses median absolute 
#       deviation (MAD) to detect outliers. If a mask’s anomaly index exceeds 2.0,
#       it is flagged as a potential backdoor indicator.
# ------------------------------------------------------------------------------

import os      # For file and directory operations.
import sys     # For accessing system-specific parameters.
import time    # For measuring execution time.
import numpy as np  # For numerical operations.

##############################
#        PARAMETERS          #
##############################


# Filename template for the recovered mask files.
# The template expects a type identifier (here, 'mask') and a label number.
FILE_FILENAME_TEMPLATE = 'tabular_visualize_%s_label_%d.csv'


##############################
#      END PARAMETERS        #
##############################


def outlier_detection(l1_norm_list, idx_mapping):
    """
    Analyze the list of L1 norms of the recovered masks to detect outliers.
    
    Outliers are determined by comparing each mask's L1 norm to the median using
    the median absolute deviation (MAD). A mask is flagged if its anomaly index
    (i.e. the normalized distance from the median) exceeds 2.0.
    
    Args:
        l1_norm_list (list of float): List of L1 norms, one per recovered mask.
        idx_mapping (dict): Mapping from label to index in l1_norm_list.
    
    Prints:
        - The list of L1 norms.
        - The median and MAD.
        - The anomaly index for each label.
        - A sorted list of flagged labels with their corresponding L1 norms.
    """
    print("check input l1-norm: ", l1_norm_list)
    consistency_constant = 1.4826  # Scaling factor for MAD under normal distribution.
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index (for minimum L1 norm): %f' % min_mad)

    # create a dictionary for labels and their corresponding anomaly indices
    anomaly_dict = {}
    flag_list = []  # List to collect labels that are flagged as outliers.
    for y_label in idx_mapping:
        # Compute anomaly index for each mask.
        # add a small epsilon value to avoid division by zero if mad value is close to zero
        epsilon_value = 1e-6
        anomaly_index = np.abs(l1_norm_list[idx_mapping[y_label]] - median) / (mad + epsilon_value)
        anomaly_dict[y_label] = anomaly_index
        print("label: ", y_label, "l1-norm: ", l1_norm_list[idx_mapping[y_label]], 
              "anomaly_index: ", anomaly_index)
        # Consider only masks with a norm below the median.
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if anomaly_index > 2.0:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: %s' %
          ', '.join(['%d: %2f' % (y_label, l_norm)
                     for y_label, l_norm in flag_list]))
    
    return anomaly_dict



def analyze_pattern_norm_dist(results_dir, num_classes):
    """
    Analyze the distribution of L1 norms of the recovered mask files.
    
    This function:
      - Iterates over each class label.
      - Loads the corresponding mask CSV file if it exists.
      - Flattens the mask to a 1D vector (if not already).
      - Computes the L1 norm (sum of absolute values) for each mask.
      - Calls outlier_detection() to analyze the distribution and flag potential outliers.
    """
    mask_list = []   # List to store each mask as a 1D numpy array.
    idx_mapping = {} # Map from label to index in mask_list.

    for y_label in range(num_classes):
        # Construct the filename for the mask using the template.
        mask_filename = FILE_FILENAME_TEMPLATE % ('mask', y_label)
        full_path = os.path.join(results_dir, mask_filename)
        if os.path.isfile(full_path):
            # Load the mask from the CSV file.
            mask = np.loadtxt(full_path, delimiter=",")
            mask = mask.flatten()  # Ensure it is a 1D vector.
            mask_list.append(mask)
            idx_mapping[y_label] = len(mask_list) - 1

    # Compute the L1 norm for each mask.
    l1_norm_list = [np.sum(np.abs(m)) for m in mask_list]

    print('%d labels found' % len(l1_norm_list))
    print("check idx_mapping", idx_mapping)

    # Analyze the L1 norm distribution to detect outliers.
    anomaly_dict = outlier_detection(l1_norm_list, idx_mapping)
    return anomaly_dict


# if __name__ == '__main__':
#     print('%s start' % sys.argv[0])
#     start_time = time.time()
#     analyze_pattern_norm_dist()
#     elapsed_time = time.time() - start_time
#     print('elapsed time %.2f s' % elapsed_time)
