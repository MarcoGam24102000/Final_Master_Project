# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:26:59 2024

@author: Rui Pinto
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def compute_contrast(directory, end_image_number):
    # Initialize an empty list to store contrast values for each video
    all_contrast_values = [] 
    all_mean_values = []
    all_std_values = []
    
    video_counter = 0
    
    list_video_numbers = [1, 5, 10, 16]

    # Iterate over all folders in the given directory
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            if "Image_Processing" in folder:
                video_counter += 1
                if video_counter in list_video_numbers:  # Process only the specified videos
                    ip_folder_path = os.path.join(root, folder)

                    for subroot, subdirs, _ in os.walk(ip_folder_path):
                        for subfolder in subdirs:
                            if "modRoisFirstVideo__080224" in subfolder:
                                subfolder_path = os.path.join(subroot, subfolder)

                                if any(os.listdir(subfolder_path)):
                                    main_sub_folder = subfolder_path 
                                    image_paths = [os.path.join(main_sub_folder, f'roi_image{i}.jpg')
                                                   for i in range(end_image_number)]

                                    if image_paths:
                                        contrast_values = []  # Initialize an empty list to store contrast values for the current video
                                        mean_values = []
                                        std_values = []
                                        for i in range(1, end_image_number):
                                            imgs_data = [plt.imread(image_path) for image_path in image_paths[:i]]
                                            imgs_mean = np.mean(imgs_data, axis=0)
                                            imgs_std = np.std(imgs_data, axis=0)
                                            
                                            mean = np.mean(imgs_mean)
                                            std = np.mean(imgs_std)
                                            
                                            try:
                                                contrast = np.mean(imgs_std / imgs_mean)
                                            except RuntimeWarning as e:
                                                print("A RuntimeWarning occurred:", e)
                                                contrast = 0  # Assign a default value when encountering NaN or other issues

                                            # Handle NaN values
                                            if np.isnan(contrast):
                                                print("NaN value detected, assigning default value")
                                                contrast = 0

                                            contrast_values.append(contrast)
                                            mean_values.append(mean)
                                            std_values.append(std)

                    # Append the contrast values for the current video to the list of all contrast values
                    all_contrast_values.append(contrast_values)
                    all_mean_values.append(mean_values)
                    all_std_values.append(std_values)
    

    # Plotting
    
    peaks_overall = []
    vals_peaks = []
    
    fig, axes = plt.subplots(2, 2)
    
    for i, mean_values in enumerate(all_mean_values):
        
        axes[0,0].plot(range(2, end_image_number + 1), mean_values, label=f'Video ' + str(list_video_numbers[i]))
      
 
    axes[0,0].set_xlabel('End Image Number \n')  
    axes[0,0].set_ylabel('Mean Value')
    axes[0,0].set_title(f'Mean Values from Start Image \n to End Image for Selected Videos', fontsize=9)
 #   axes[0,0].legend(loc='upper left')
  #  axes[0,0].show() 
    
    for i, std_values in enumerate(all_std_values):
        axes[0,1].plot(range(2, end_image_number + 1), std_values, label=f'Video ' + str(list_video_numbers[i]))
                
#    plt.figure(2)
    axes[0,1].set_xlabel('End Image Number \n')  
    axes[0,1].set_ylabel('STD Value')
    axes[0,1].set_title(f'STD Values from Start Image \n to End Image for Selected Videos', fontsize=9)
#    axes[0,1].legend(loc='upper right')
 #   axes[0,1].show()
        
    for i, contrast_values in enumerate(all_contrast_values):
        
        print("One: " + str(len(range(2, 41 + 1))))
        print("Two: " + str(len(contrast_values[:40])))
        
        peaks_inter_phase = []
        val_peaks = []
        
        for indVal, val in enumerate(contrast_values[:40]):
            if indVal > 0:
                if val > contrast_values[indVal-1] and val > contrast_values[indVal+1]:
                    peak_found = indVal
                    peaks_inter_phase.append(peak_found)
                    val_peaks.append(val)
                    
        peaks_overall.append(peaks_inter_phase)
        vals_peaks.append(val_peaks)
        
  ##      plt.plot(range(2, 41 + 1), contrast_values[:40], label=f'Video ' + str(list_video_numbers[i]))
        
       
        axes[1,0].plot(range(2, end_image_number + 1), contrast_values, label=f'Video ' + str(list_video_numbers[i]))
                
  #  plt.figure(3)
    axes[1,0].set_xlabel('End Image Number')  
    axes[1,0].set_ylabel('Mean Contrast')
    axes[1,0].set_title(f'Mean Contrast Values from Start Image \n to End Image for Selected Videos', fontsize=9)
 #   axes[1,0].legend(loc='lower left')
 #   axes[1,0].show()
    
    for indN, peakList in enumerate(peaks_overall):
        axes[1,1].scatter(peakList, vals_peaks[indN], label=f'Peak analysis for video ' + str(list_video_numbers[indN]))
    
  #  plt.figure(4)
    axes[1,1].set_xlabel('Image index')  
    axes[1,1].set_ylabel('Peak')
    axes[1,1].set_title(f'Peak analysis until \n image 40', fontsize=9)
 #   axes[1,1].legend(loc='lower right')
 #   axes[1,1].show()
 
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=5)
 
    plt.tight_layout()
    
    return peaks_overall 

# Example usage
peaks_overall = compute_contrast("C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/Acq_23_1_24_pseudomonas", 126)

peaks_flatten = []
for p in peaks_overall:
    for single_p in p:
        peaks_flatten.append(single_p)

peaksNew = []

for p in peaks_flatten:
    if p not in peaksNew:
        peaksNew.append(p)

peaksNew.sort()

peaksNewCleaned = []

for p in peaksNew:
    if len(peaksNewCleaned) > 0:
        meanComputed = False
        for indX, x in enumerate(peaksNewCleaned):
            if abs(p-x) < 3:
                meanX = int((p+x)/2)
                peaksNewCleaned[indX] = meanX
                meanComputed = True
        if not meanComputed:
            peaksNewCleaned.append(p)
    else:
        peaksNewCleaned.append(p)
    
        
        
