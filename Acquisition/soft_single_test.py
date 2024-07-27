# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 00:05:15 2022

@author: marco
"""

def videoAnalysis_single(tupleForProcessing):    
    
    decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, pathRoiStart, pathRoiEnd, first_clustering_storing_output, pathPythonFile = tupleForProcessing

    import time
    startTime = time.time() 
    
    #%%
    
    import math
    import os
    import cv2
    import imageio
    import shutil
    import sys 
    import subprocess
    import shlex
    import numpy as np
    import pandas as pd
    import xlsxwriter
    import configparser
    
    sobel_error = True
    
    while sobel_error == True:
        try:    
            from skimage.filters import sobel, scharr, gaussian, median, roberts, laplace, hessian, frangi, prewitt, sobel_h, sobel_v
        except ModuleNotFoundError:
            sobel_error = True
                 
        else:
            sobel_error = False 
            
    import matplotlib.pyplot as plt  
    import skimage.morphology as morphology
    import scipy.ndimage as ndi 
    from skimage.color import label2rgb  
    from skimage.segmentation import watershed
    ## from _watershed import watershed
    from sklearn.model_selection import train_test_split  
    from sklearn.cluster import KMeans  
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.decomposition import PCA  
    from scipy.spatial.distance import pdist, squareform  
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    import csv
    from spyder_kernels.utils.iofuncs import load_dictionary  
    from spyder_kernels.utils.iofuncs import save_dictionary 
    import pickle
    import tarfile
    from sewar.full_ref import mse, rmse, _rmse_sw_single, rmse_sw, psnr, _uqi_single, uqi, ssim, ergas, scc, rase, sam, msssim, vifp, psnrb; 
    
    ######################
    ######################
    ######################
    
    data_results = []
    
    ######################
    ######################
    ######################
    
    ## parent_dir = 'C:/Research/'
    parent_dir = dest_path    
                                                    
    infoI_seg = []
    infoJ_seg = []
    
    count = 0
    
    #%% Apart from the code shown above 
    
   ## pre_dir = 'C:/Research/VideosAlmostLaserSpeckle/'
    
    pre_dir = mtsVideoPath  
    
    #%%
    
    ## IFVP = 'C:/Research/DataSequence_'
    folder = IFVP + str(sequence_name) + '/'
        
    ##locationMP4_file = 'FilesFor_6_8_X1_'
    
    name_folder = locationMP4_file + str(sequence_name)
    
    pre_dirInit = "C:\\Users\\Other\\files_python\\py_scripts\\ffmpeg-5.0.1-full_build\\bin\\GUI\\"
    
    path_in = pre_dirInit + "test_" + "00" + str(sequence_name) + ".avi"    ## .mts 
    
    mp4_initDir = os.path.join(mp4VideoFile + name_folder + "\\")   
    
    endInd = 0
     
    while True:
        endInd += 1
        if os.path.exists(mp4_initDir):
            mp4_initDirx = mp4_initDir[:-1]
            if mp4_initDirx[-2] == '_':
                mp4_initDirxt = mp4_initDirx[:-1] 
                mp4_initDirx = mp4_initDirxt + str(endInd) + '_'                
            elif mp4_initDirx[-3] == '_':
                mp4_initDirxt = mp4_initDirx[:-2] 
                mp4_initDirx = mp4_initDirxt + str(endInd) + '_'                
            
            mp4_initDir = mp4_initDirx + "\\"           
            
        else:
            break
    
    os.mkdir(mp4_initDir) 
    
    path_out = mp4_initDir + "test_" + "00" + str(sequence_name) + '.mp4'
    path_out_mp4 = pre_dirInit + "test_" + "00" + str(sequence_name) + ".mp4"    ## .mts     

    def pairedNumber(n1, div):
        
        if not (n1%2 == 0): 
            a = int(n1/2)
             
            if n1 > a*2:
                n2 = a*2
            else: 
                n2 = n1       
            
            return n2 
        return n1    
     
    
    def loadMP4_file(path_in, path_out):
                name_in = "test_" + "00" + str(sequence_name) + ".avi"
                name_out = "test_" + "00" + str(sequence_name) + ".mp4"
                cmd = 'ffmpeg -i  ""C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/' + name_in + "" + ' "" C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/' + name_out + ""
                print("ffmpeg command: " + cmd)
                
                cmd_arr = shlex.split(cmd)
                cmd_arr = np.array([cmd_arr])
                cmd_arr = np.delete(cmd_arr, 3)
                cmd = np.array([cmd_arr])[0].tolist() 
                
                subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=False)
                dir_out = path_out            
                time.sleep(5) 
                return dir_out 
    
    print("Path:    " + pre_dir+name_folder) 
    
    endInd = 0
    
    while True:
    
        if os.path.isdir(pre_dir+name_folder): 
            
            endInd += 1
            
            if name_folder[-2] == '_':  
                
                name_folderx = name_folder[:-1] 
                name_folder = name_folderx + str(endInd)
            elif name_folder[-3] == '_':
            
                name_folderx = name_folder[:-2]
                name_folder = name_folderx + str(endInd)
        else:
            break
    
    if os.path.isdir(pre_dir+name_folder): 
        print("Directory already exists !!!")
    else: 
    
        newPath = os.path.join(pre_dir, name_folder)    
        os.mkdir(newPath)
        
        print("Directory created")
     
         
        if os.path.isfile('path_out' + '/' + str(sequence_name) + '.mp4'):
            print("MP4 File already exists inside directory") 
        else:
            print(path_in)
            print(path_out)
            src_dir = loadMP4_file(path_in, path_out_mp4)           
            time.sleep(100)           
            os.rename(path_out_mp4, path_out) 
            while(os.path.exists(path_out) == False): 
                time.sleep(5)
            print("MP4 file loaded")          
        
        time.sleep(100)    
        
        src_dir = path_out 
        
        print("Source Directory for MP4 file: " + src_dir)
        
        vidcap = cv2.VideoCapture(src_dir) 
        
        fps_out = 50    
        index_in = -1
        index_out = -1    
        reader = imageio.get_reader(src_dir)
        fps_in = reader.get_meta_data()['fps']
        
        count = 0    
       
        while(True): 
            
            success, image = vidcap.read()
            if success:
                
                if os.path.isdir(IFVP + str(sequence_name) + "_2"):
                    print("Directory already exists !!!")
                
                else:                     
                    newPath = os.path.join(IFVP + str(sequence_name) + "_2") 
                    os.mkdir(newPath)     
                    print("Another directory created")
                 
                cv2.imwrite(IFVP + str(sequence_name) + "_2" + "/video_image%d.jpg" % count, image)     
                count += 1
            else:  
                break           
    
        print("Images loaded") 
        
    totCountImages = count
        
    #%%
    
    roi_path = os.path.join(mainPathVideoData + roiPath + str(sequence_name) + '/')    
    
    endInd = 0
    
    while True:
        endInd += 1
        if os.path.exists(roi_path):
            roi_pathx = roi_path[:-1]
            if roi_pathx[-2] == '_':
               roi_pathxt = roi_pathx[:-1] 
               roi_pathx = roi_pathxt + str(endInd) + '_'                
            elif roi_pathx[-3] == '_':
                roi_pathxt = roi_pathx[:-2] 
                roi_pathx = roi_pathxt + str(endInd) + '_'                
            
            roi_path = roi_pathx + '/'          
            
        else:
            break   
    
    os.mkdir(roi_path)  

 
    dec_thresh = decisorLevel     
    
    for video_image in range(0,count):        
        
            print("Analysing for image " + str(video_image) + " th") 
        
            image = cv2.imread(IFVP + str(sequence_name) + "_2" + "/video_image%d.jpg" % video_image)        
            imx = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)         
     
            elevation_map = sobel(imx)    
            
            cv2.imwrite(roi_path + "/elevation_map_%d.jpg" % video_image, elevation_map)
            plt.imsave('el_map_' + str(video_image) + '.png', elevation_map, cmap=plt.cm.gray)   ##  interpolation='nearest'
            
            markers = np.zeros_like(imx)        
            markers[imx < 50] = 1         ## 100     
            markers[imx > 100] = 2         ## 150    
            
            segmentation = watershed(elevation_map, markers)        
            segmentation = ndi.binary_fill_holes(segmentation - 1)       
            labeled_regions, _ = ndi.label(segmentation)         
            image_label_overlay = label2rgb(labeled_regions, image=imx)
            
            print("Lenght third dimension of image overlay: " + "(" + str(len(image_label_overlay)), str(len(image_label_overlay[0])), str(len(image_label_overlay[1])), str(len(image_label_overlay[2])) + ")")
            
            imageOverlay = image_label_overlay[:,:,0]
            
            cv2.imwrite(roi_path + "/overlay_image_bef_selection%d.jpg" % video_image, imageOverlay)             
            
            i_essential = []
            j_essential = []        
            
            imageOverlayDen = np.zeros((1080,1920))
            
            print("Estimating image overlay ... ")
            
            for j in range(0,len(imageOverlay[0])):
                for i in range(0,len(imageOverlay)):
                    if imageOverlay[i,j] >= dec_thresh:         ### ------------------------
                        print("Above threshold")
                        imageOverlayDen[i,j] = 255
                        i_essential.append(i)
                        j_essential.append(j)
                    else:
                        print("Under threshold")
                        imageOverlayDen[i,j] = 0
                        
            i_un = np.array([np.unique(np.array([i_essential]))])
            j_un = np.array([np.unique(np.array([j_essential]))])
            i_arr = np.zeros((1,len(i_un)))
            j_arr = np.zeros((1,len(j_un))) 
            
            min_i = np.min(i_un)
            max_i = np.max(i_un)
            min_j = np.min(j_un)
            max_j = np.max(j_un)
            
            if True:         
              roi_imx = imx[min_i:max_i, min_j:max_j] 
              
              infoI_seg.append((min_i, max_i))
              infoJ_seg.append((min_j, max_j))
            
            cv2.imwrite(roi_path + "/overlay_image%d.jpg" % video_image, imageOverlayDen)   
            cv2.imwrite(roi_path + "/roi_image%d.jpg" % video_image, roi_imx)  
            
    infoI_seg = []
    infoJ_seg = []
    
    count = 0
    
    fixed_roi_path = os.path.join(mainPathVideoData + newRoiPath + str(sequence_name) + '/')

    endInd = 0
    
    while True:
        endInd += 1
        if os.path.exists(fixed_roi_path):
            fixed_roi_pathx = fixed_roi_path[:-1]
            if fixed_roi_pathx[-2] == '_':
               fixed_roi_pathxt = fixed_roi_pathx[:-1] 
               fixed_roi_pathx = fixed_roi_pathxt + str(endInd) + '_'                
            elif roi_pathx[-3] == '_':
                fixed_roi_pathxt = fixed_roi_pathx[:-2] 
                fixed_roi_pathx = fixed_roi_pathxt + str(endInd) + '_'                
            
            fixed_roi_path = fixed_roi_pathx + '/'          
            
        else:
            break   
    
    os.mkdir(fixed_roi_path)
    
    ## fixed_roi_path = 'C:/Research/Approach317_new'  
    
    for video_image in range(0,totCountImages):  
        
            print("Analysing for roi " + str(video_image) + " th")
        
            image = cv2.imread(roi_path + "/roi_image%d.jpg"  % video_image)        
            imx = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
            
            x_dim = len(imx)
            y_dim = len(imx[0])        
            x_center = int(x_dim/2) 
            y_center = int(y_dim/2)        
            imd = imx[x_center-50:x_center+50, y_center-50:y_center+50]
            
            cv2.imwrite(fixed_roi_path + "/roi_image%d.jpg" % video_image, imd)         
            count += 1 
    
    #%%
    
    roi_bef = pathRoiStart + str(sequence_name)
    roi_after = pathRoiEnd + str(sequence_name)    
    
    #%%
    
    start_mark_before_inter = 0
    end_mark_before_inter = 50
     
    numberImagesBef = end_mark_before_inter - start_mark_before_inter + 1 
     
    start_mark_after_inter = count-50
    end_mark_after_inter = count
     
    numberImagesAfter = end_mark_after_inter - start_mark_after_inter + 1
     
    newPathA = os.path.join(roi_bef) 
    if newPathA[0] != 'C':
        newPathA = 'C' + newPathA
    
    endInd = 0
    
    while True:
        
        endInd += 1
        
        if os.path.exists(newPathA + '/'):
            if newPathA[-2] == 'f':
                newPathAx = newPathA[:-1]
            elif newPathA[-3] == 'f':
                newPathAx = newPathA[:-2]
            
            newPathA = newPathAx + str(endInd)
        else:
            break
            
    os.mkdir(newPathA)   
     
    newPathB = os.path.join(roi_after)     
    if newPathB[0] != 'C':
        newPathB = 'C' + newPathB
        
    endInd = 0
    
    while True:
        
        endInd += 1
        
        if os.path.exists(newPathB + '/'):
            if newPathB[-2] == 'r':
                newPathBx = newPathB[:-1]
            elif newPathB[-3] == 'r':
                newPathBx = newPathB[:-2]
            
            newPathB = newPathBx + str(endInd)
        else:
            break
        
    os.mkdir(newPathB)  
     
    ind_bef = 0 
    ind_after = 0
    
    print("Count: " + str(count))
     
    for i in range(0, count):
        print("Writing splitted, for image " + str(i))
        imx = cv2.imread(fixed_roi_path + "/roi_image" + str(i) + ".jpg")
#        print(imx)
        imxa = cv2.cvtColor(imx, cv2.COLOR_BGR2GRAY) 
         
        if i >= start_mark_before_inter and i < int(count/2):
            print("Before at " + str(ind_bef) + " image")
            
            if roi_bef[0] != 'C':
                roi_bef = 'C' + roi_bef
                
            print(roi_bef + "/roi_image")
            
            cv2.imwrite(roi_bef + "/roi_image%d.jpg" % ind_bef, imxa)
            ind_bef += 1    
        else:
            if i >= start_mark_after_inter and i < end_mark_after_inter:
                print("After at " + str(ind_after) + " image")               
                
                if roi_after[0] != 'C':
                    roi_after = 'C' + roi_after
                    
                print(roi_after + "/roi_image")
                
                cv2.imwrite(roi_after + "/roi_image%d.jpg" % ind_after, imxa)
                ind_after += 1
                
    labelsFeatQuality = ['MSE', 'RMSE', 'RMSE_SINGLE', 'RMSE_SW', 'PSNR', 'UQI_SINGLE', 
                         'UQI', 'SSIM', 'ERGAS', 'SCC', 
                         'RASE', 'SAM', 'MSSSIM', 'VIFP', 'PSNRB']
    
    labelsFeatQualityValues = ['MSE', 'RMSE', 'RMSE_SINGLE', 'RMSE_SW', 'PSNR', 'UQI_SINGLE', 
                         'UQI', 'SSIMS', 'CSS', 'ERGAS', 'SCC', 
                         'RASE', 'SAM', 'MSSSIM', 'VIFP', 'PSNRB']
     
    anotherPreviousMetrics = ["Mean", "STD", "Contrast", "ASM", "Max"] 
    
    for ind_an, an in enumerate(anotherPreviousMetrics):
        labelsFeatQualityValues += [str(an)]
        
    #%%
        
    bigCoupleImages = []
    
    def findAUC(conf_matrix):
        TA = conf_matrix[0,0]
        TB = conf_matrix[1,1]
        TC = conf_matrix[2,2]
        FA1 = conf_matrix[0,1]
        FA2 = conf_matrix[0,2]
        FB1 = conf_matrix[1,0]
        FB2 = conf_matrix[1,2]
        FC1 = conf_matrix[2,0]
        FC2 = conf_matrix[2,1]    
        
        T = TA+TB+TC    
        AC = (T/(T+FA1+FA2+FB1+FB2+FC1+FC2))
        
        return AC  
    
    def meanImages(image_1, image_2):
       
        meanImage = np.zeros((len(image_1), len(image_1[0])))
        
        for j in range(0,len(image_1[0])):
            for i in range(0, len(image_1)): 
                meanImage[i,j] = int((image_2[i,j]+image_1[i,j])/2)
        sumMeanImages = np.sum(meanImage)       
        
        return sumMeanImages
    
    def stdImages(image_1, image_2):
       
        stdImage = np.zeros((len(image_1), len(image_1[0])))
        
        for j in range(0,len(image_1[0])): 
            for i in range(0, len(image_1)): 
                stdImage[i,j] = int(abs((int(image_2[i,j])-image_1[i,j])))
        sumSTD_images = np.sum(stdImage)       
        
        return sumSTD_images 
    
    def contrastImages(image_1, image_2):
        
        meanImage = np.zeros((len(image_1), len(image_1[0])))
        
        for j in range(0,len(image_1[0])):
            for i in range(0, len(image_1)): 
                meanImage[i,j] = int((image_2[i,j]+image_1[i,j])/2)
        
        stdImage = np.zeros((len(image_1), len(image_1[0]))) 
                
        for j in range(0,len(image_1[0])): 
            for i in range(0, len(image_1)): 
                stdImage[i,j] = int(abs((image_2[i,j]-image_1[i,j])))
        
        contrastImage = np.zeros((len(image_1), len(image_1[0]))) 
        
        for j in range(0,len(image_1[0])): 
            for i in range(0, len(image_1)): 
                if meanImage[i,j] != 0:
                    contrastImage[i,j] = stdImage[i,j]/meanImage[i,j]          
        
        contrastImageSum = np.sum(contrastImage)
        
        return contrastImageSum
        
    def asmImages(image_1, image_2): 
        
        stdImage = np.zeros((len(image_1), len(image_1[0])))
        asmImageSum = 0
                
        for j in range(0,len(image_1[0])): 
            for i in range(0, len(image_1)): 
                stdImage[i,j] = int(abs((image_2[i,j]-image_1[i,j])))
        
        for j in range(0,len(image_1[0])): 
            for i in range(0, len(image_1)):             
                asmImageSum += stdImage[i,j] ** 2            
        
        return asmImageSum 
    
    def maxImage(image_1, image_2):
        maxValueWithImages = 0
        maxForImage_A = 0
        for j in range(0,len(image_1[0])): 
            for i in range(0, len(image_1)):
                maxForImage_A += image_1[i,j]
        
        maxForImage_B = 0
        for j in range(0,len(image_2[0])): 
            for i in range(0, len(image_2)):
                maxForImage_B += image_2[i,j]
                
        if maxForImage_A >= maxForImage_B: 
            maxValueWithImages = maxForImage_A 
        else:
            maxValueWithImages = maxForImage_B
            
        return maxValueWithImages          
    
    folder = fixed_roi_path + '/'
    couple_images = []  
    
    for ind in range(0,49):
        print("Reading before and after-related images: " + str(ind+1) + " th image ...")
        im1 = cv2.imread(roi_bef + "/roi_image%d.jpg" % ind)
        im2 = cv2.imread(roi_after + "/roi_image%d.jpg" % ind)
        
        ori_img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        ori_img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)  
       
        couple_images = [ori_img1, ori_img2]               
               
        bigCoupleImages.append(couple_images)
        couple_images = []
    
    #%%    
            
    setMetrics = [] 
    setMetricsValues = []
    mseList = []
    rmseList = []
    rmseSWList = []
    psnrList = [] 
    uqiSingleList = []
    uqiList = []
    ssimsList = []
    cssList = []
    ergasList = []
    sccList = []
    raseList = []
    samList = []
    msssimList = []
    vifpList = []
    psnrbList = []
    meanList = []
    stdList = []
    contrastList = []
    asmList = []
    maxList = []            
    
    mseList.insert(0, labelsFeatQualityValues[0])
    rmseList.insert(0, labelsFeatQualityValues[1])
    rmseSWList.insert(0, labelsFeatQualityValues[2])
    psnrList.insert(0, labelsFeatQualityValues[3])
    uqiSingleList.insert(0, labelsFeatQualityValues[4])
    uqiList.insert(0, labelsFeatQualityValues[5])
    ssimsList.insert(0, labelsFeatQualityValues[6]) 
    cssList.insert(0, labelsFeatQualityValues[7])
    ergasList.insert(0, labelsFeatQualityValues[8])
    sccList.insert(0, labelsFeatQualityValues[9])
    raseList.insert(0, labelsFeatQualityValues[10])
    samList.insert(0, labelsFeatQualityValues[11])
    msssimList.insert(0, labelsFeatQualityValues[12])
    vifpList.insert(0, labelsFeatQualityValues[13])
    psnrbList.insert(0, labelsFeatQualityValues[14])
    meanList.insert(0, labelsFeatQualityValues[15])
    stdList.insert(0, labelsFeatQualityValues[16])
    contrastList.insert(0, labelsFeatQualityValues[17])
    asmList.insert(0, labelsFeatQualityValues[18])
    maxList.insert(0, labelsFeatQualityValues[19])
    
    setMetricsValues.append(labelsFeatQualityValues)
    
    for ind_couple, couple in enumerate(bigCoupleImages):
        print("Find metrics for couple " + str(ind_couple) + " th couple of images")
        origImage = np.array([couple])[0,0,:,:]
        deformedImage = np.array([couple])[0,1,:,:]
        
        mse_feat = mse(origImage, deformedImage)
        if math.isnan(mse_feat) or math.isinf(mse_feat): 
            mse_feat = 0 
        
        rmse_feat = rmse(origImage, deformedImage)
        if math.isnan(rmse_feat) or math.isinf(rmse_feat):
            rmse_feat = 0 
         
        rmse_single_feat = _rmse_sw_single(origImage, deformedImage,50)
        value_rmse, matrix_errors = rmse_single_feat
        if math.isnan(value_rmse) or math.isinf(value_rmse): 
            value_rmse = 0
            rmse_single_feat = (value_rmse, matrix_errors)
        
        rmse_sw_feat = rmse_sw(origImage, deformedImage)
        value_rmse_sw, matrix_errors_sw = rmse_sw_feat
        if math.isnan(value_rmse_sw) or math.isinf(value_rmse_sw): 
            value_rmse_sw = 0
            rmse_sw_feat = (value_rmse_sw, matrix_errors_sw)
        
        psnr_feat =  psnr(origImage, deformedImage)
        if math.isnan(psnr_feat) or math.isinf(psnr_feat): 
            psnr_feat = 0        
        
        uqi_single_feat = _uqi_single(origImage, deformedImage,50)
        if math.isnan(uqi_single_feat) or math.isinf(uqi_single_feat):  
            uqi_single_feat = 0
        
        uqi_feat = uqi(origImage, deformedImage)
        if math.isnan(uqi_feat) or math.isinf(uqi_feat):   
            uqi_feat = 0 
        
        ssim_feat = ssim(origImage, deformedImage)  
        ssims, css = ssim_feat
        if math.isnan(ssims) and math.isnan(css):
            ssims = 0
            css = 0 
            ssim_feat = (ssims, css) 
        else:
            if math.isnan(ssims) and not math.isnan(css):
                ssims = 0
                ssim_feat = (ssims, css)
            else:
                if math.isnan(css) and not math.isnan(ssims): 
                    css = 0 
                    ssim_feat = (ssims, css)
        
        if math.isinf(ssims) and math.isinf(css):
                ssims = 0
                css = 0 
                ssim_feat = (ssims, css)
        else:
                if math.isinf(ssims) and not math.isinf(css):
                    ssims = 0
                    ssim_feat = (ssims, css)
                else:
                    if math.isinf(css) and not math.isinf(ssims): 
                        css = 0 
                        ssim_feat = (ssims, css)
                    
        ergas_feat = ergas(origImage, deformedImage) 
        if math.isnan(ergas_feat) or math.isinf(ergas_feat):
            ergas_feat = 0
        
        scc_feat = scc(origImage, deformedImage) 
        if math.isnan(scc_feat) or math.isinf(scc_feat):
            scc_feat = 0
        
        rase_feat =  rase(origImage, deformedImage)
        if math.isnan(rase_feat) or math.isinf(rase_feat):
            rase_feat = 0
        
        sam_feat = sam(origImage, deformedImage)  
        if math.isnan(sam_feat) or math.isinf(sam_feat):
            sam_feat = 0
        
        msssim_feat = msssim(origImage, deformedImage)
        if math.isnan(msssim_feat.real) or math.isinf(msssim_feat.real):
            msssim_feat = 0+0j
        
        vifp_feat = vifp(origImage, deformedImage)  
        if math.isnan(vifp_feat) or math.isinf(vifp_feat):
            vifp_feat = 0
        
        psnrb_feat = psnrb(origImage, deformedImage) 
        if math.isnan(psnrb_feat) or math.isinf(psnrb_feat):
            psnrb_feat = 0 
            
        mean_feat = meanImages(origImage, deformedImage)    
        std_feat = stdImages(origImage, deformedImage)
        contrastFeat = contrastImages(origImage, deformedImage)
        asmFeat = asmImages(origImage, deformedImage)
        maxFeat = maxImage(origImage, deformedImage)  
         
        metrics = [mse_feat, rmse_feat, rmse_single_feat, rmse_sw_feat, psnr_feat,
                   uqi_single_feat, uqi_feat, ssim_feat, ergas_feat, scc_feat,
                   rase_feat, sam_feat, msssim_feat, vifp_feat, psnrb_feat]
        
        metrics_values = [mse_feat, rmse_feat, value_rmse, value_rmse_sw, psnr_feat,
                   uqi_single_feat, uqi_feat, ssims, css, ergas_feat, scc_feat,
                   rase_feat, sam_feat, msssim_feat.real, vifp_feat, psnrb_feat]
        
        newMetricsValues = [mean_feat, std_feat, contrastFeat, asmFeat, maxFeat] 
        
        for ind_newM, newM in enumerate(newMetricsValues):    
           metrics_values.append(newM)
        
        mseList.append(mse_feat)
        rmseList.append(rmse_feat)
        rmseSWList.append(value_rmse_sw)
        psnrList.append(psnr_feat) 
        uqiSingleList.append(uqi_single_feat)
        uqiList.append(uqi_feat)
        ssimsList.append(ssims)
        cssList.append(css)
        ergasList.append(ergas_feat) 
        sccList.append(scc_feat)
        raseList.append(rase_feat)
        samList.append(sam_feat)
        msssimList.append(msssim_feat.real)
        vifpList.append(vifp_feat)
        psnrbList.append(psnrb_feat)   
        meanList.append(mean_feat)
        stdList.append(std_feat)
        contrastList.append(contrastFeat)
        asmList.append(asmFeat)
        maxList.append(maxFeat)
         
        setMetrics.append(metrics)
        setMetricsValues.append(metrics_values)
        
    newListMetrics = []
    
    for ind,  metricValue in enumerate(setMetricsValues):
        number_No_zeros = 0
        number_No_zeros = np.count_nonzero(np.array([metricValue]))
        
        if int(number_No_zeros) > 12:   
            newListMetrics.append(metricValue) 
    
    metric_1 = []
    metric_2 = []
    metric_3 = []
    metric_4 = []
    metric_5 = []
    metric_6 = []
    metric_7 = []
    metric_8 = []
    metric_9 = []
    metric_10 = []
    metric_11 = []
    metric_12 = []
    metric_13 = []
    metric_14 = []
    metric_15 = []
    metric_16 = []
    metric_17 = []  
    metric_18 = [] 
    metric_19 = []
    metric_20 = []
    metric_21 = []  
    
    not_constant_Flag = 0
            
    for ind_new, metricNew in enumerate(newListMetrics):
      
            for indInsideMetric, valueInMetric in enumerate(metricNew):
               
                if ind_new > 0:
                
                    if indInsideMetric == 0: metric_1.append(valueInMetric)
                    if indInsideMetric == 1: metric_2.append(valueInMetric)
                    if indInsideMetric == 2: metric_3.append(valueInMetric)
                    if indInsideMetric == 3: metric_4.append(valueInMetric)
                    if indInsideMetric == 4: metric_5.append(valueInMetric)
                    if indInsideMetric == 5: metric_6.append(valueInMetric)
                    if indInsideMetric == 6: metric_7.append(valueInMetric)
                    if indInsideMetric == 7: metric_8.append(valueInMetric)
                    if indInsideMetric == 8: metric_9.append(valueInMetric)
                    if indInsideMetric == 9: metric_10.append(valueInMetric)
                    if indInsideMetric == 10: metric_11.append(valueInMetric)
                    if indInsideMetric == 11: metric_12.append(valueInMetric)
                    if indInsideMetric == 12: metric_13.append(valueInMetric)
                    if indInsideMetric == 13: metric_14.append(valueInMetric)
                    if indInsideMetric == 14: metric_15.append(valueInMetric) 
                    if indInsideMetric == 15: metric_16.append(valueInMetric)
                    if indInsideMetric == 16: metric_17.append(valueInMetric)
                    if indInsideMetric == 17: metric_18.append(valueInMetric)
                    if indInsideMetric == 18: metric_19.append(valueInMetric)
                    if indInsideMetric == 19: metric_20.append(valueInMetric)
                    if indInsideMetric == 20: metric_21.append(valueInMetric)
                    
                
    listMetricsForEvaluation = [metric_1, metric_2, metric_3, metric_4, metric_5, metric_6, metric_7, metric_8, metric_9, metric_10, metric_11, metric_12, metric_13, metric_14, metric_15, metric_16, metric_17, metric_18, metric_19, metric_20, metric_21]
           
    for ind_metricConstant, newMetricEval in enumerate(listMetricsForEvaluation):        
                if len(np.unique(np.array([newMetricEval]))) == 1:
                    print("Constant feature detected")              
                    not_constant_Flag += 1           
                    listMetricsForEvaluation.pop(ind_metricConstant)               
    
    secRoundNewListMetrics = []
    metricsIndicesDeleted = []
    
    number_metrics = 20
                
    if True:
        for ind_new_sec, newMetricEval in enumerate(listMetricsForEvaluation):
            number_No_zeros = 0
            number_No_zeros = np.count_nonzero(np.array([newMetricEval]))
            
            if number_No_zeros > len(newMetricEval)/2:
                secRoundNewListMetrics.append(newMetricEval)
            else:
                metricsIndicesDeleted.append(ind_new_sec)
    
    test_size = 0.2
    newsecRoundNewListMetrics = []
    
    for metricSec in secRoundNewListMetrics:
        lenMax = len(metricSec)   
        
    if lenMax%10 != 0:
       lenMax = round(lenMax/10)*10-10
     
    for metricSec in secRoundNewListMetrics:   
        metricSec = metricSec[0:lenMax]
        newsecRoundNewListMetrics.append(metricSec) 
     
    secRoundNewListMetrics = newsecRoundNewListMetrics                        
    secRoundNewListMetricsArr = np.array([secRoundNewListMetrics])[0].T
    
    if int(number_metrics) > len(secRoundNewListMetricsArr[0]):
        number_metrics = len(secRoundNewListMetricsArr[0])
        
        
    resT = np.array([ secRoundNewListMetricsArr[:, int(number_metrics)-1]]).T 
    
    #%%
    
    train_data, test_data, labels_train_data, labels_test_data = train_test_split(secRoundNewListMetricsArr[:, 0:int(number_metrics)-1], resT, test_size =test_size, random_state = 42)
    treino_lenght = int((1-test_size)*lenMax)
    
    # Método Silhouette - análise para um nº variável de clusters   
    aux = 0 
    max_silhouette = 0
    silhouette_vector = [] 
    n_clusters3 = range(2, treino_lenght) 
    for j in n_clusters3:
        km =KMeans(n_clusters=j, max_iter=300, n_init=5).fit(train_data)
        labels3 = km.labels_
        silhouette_avg = silhouette_score(train_data, labels3)
        print("For n_clusters=", j, "The averegae silhouette_score is:", silhouette_avg)   
        aux = silhouette_avg 
        if aux > max_silhouette: 
            max_silhouette = aux
            number_recommended_clusters = j    
        silhouette_vector.append(silhouette_avg)  
        
    print("\n\nCom base no gráfico Elbow e no método Silhouette, é recomendável formar", number_recommended_clusters, "clusters!") 
    
    secRoundNewListMetrics = np.array([secRoundNewListMetrics])[0].T.tolist()
    trainListData = train_data.tolist() 
    nclusters = 6
    
    kmeans = KMeans(n_clusters=nclusters, max_iter=500, n_init=8).fit(trainListData) 
    Cluster_ID = kmeans.labels_ 
    centroides_A = kmeans.cluster_centers_   
    print("Centroides dos ",number_recommended_clusters, " clusters recomendados:\n", centroides_A)
    
    Cluster_ID_transpose = np.array([np.array([Cluster_ID]).T[0:treino_lenght,0]])
    
    objetos_c1 = []
    objetos_c2 = []
    objetos_c3 = [] 
    objetos_c4 = [] 
    objetos_c5 = [] 
    objetos_c6 = [] 
    
    for i in range (0, len(Cluster_ID_transpose[0])):
        if Cluster_ID_transpose[0,i] == 0:
            objetos_c1.append(train_data[i, :])
            i_1 = i
        elif Cluster_ID_transpose[0,i] == 1:
            objetos_c2.append(train_data[i, :]) 
            i_2 = i
        elif Cluster_ID_transpose[0,i] == 2:
            objetos_c3.append(train_data[i, :]) 
            i_3 = i
        elif Cluster_ID_transpose[0,i] == 3:
            objetos_c4.append(train_data[i, :]) 
            i_4 = i
        elif Cluster_ID_transpose[0,i] == 4:
            objetos_c5.append(train_data[i, :])
            i_5 = i
        elif Cluster_ID_transpose[0,i] == 5:
            objetos_c6.append(train_data[i, :])  
            i_6 = i     
            
    list1 = list(zip(*objetos_c1)) 
    list2 = list(zip(*objetos_c2)) 
    list3 = list(zip(*objetos_c3)) 
    list4 = list(zip(*objetos_c4)) 
    list5 = list(zip(*objetos_c5)) 
    list6 = list(zip(*objetos_c6)) 
    
    print(" -- Lists of clusters generated")
    
    for l in list1:
        LenList_1 = len(l)
    for l in list2:
        LenList_2 = len(l)
    for l in list3:
        LenList_3 = len(l)
    for l in list4:
        LenList_4 = len(l)
    for l in list5:
        LenList_5 = len(l)
    for l in list6:
        LenList_6 = len(l)    
    
    indForFolderClustering = []    
    
    list1FirstArr = np.array([np.array([list1[0]])[0]])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    singIndData = []
    
    for ind in range(0,len(list1FirstArr[0])):
       tupleIndiceImage = np.where(listArrToCompare == list1FirstArr[0,ind])
       rub, ind_data = tupleIndiceImage
       ind_data = np.array([ind_data])
       
       if len(ind_data[0]) == 1:
           singIndData.append(ind_data)
       else:
           if len(ind_data[0]) == 2:
               print("Not singular")
               singIndData.append(ind_data[0,0])
               singIndData.append(ind_data[0,1])           
    
    newSingData = []
    
    singIndData = np.array([np.unique(np.array([singIndData]))]).tolist()[0]
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))     
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd)    
          
    else:
        print("Not equal at phase 2")   
    
    list2FirstArr = np.array([np.array([list2[0]])[0]])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    singIndData = []
    
    for ind in range(0,len(list2FirstArr[0])):
       tupleIndiceImage = np.where(listArrToCompare == list2FirstArr[0,ind])
       rub, ind_data = tupleIndiceImage
       ind_data = np.array([ind_data])
       
       if len(ind_data[0]) == 1:
           singIndData.append(ind_data)
       else:
           print("Not singular")
           singIndData.append(ind_data[0,0])
           singIndData.append(ind_data[0,1])    
    
    newSingData = []
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))    
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd) 
           
    else:
        print("Not equal at phase 2")   
    
    list3FirstArr = np.array([np.array([list3[0]])[0]])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    singIndData = []
    
    for ind in range(0,len(list3FirstArr[0])):
       tupleIndiceImage = np.where(listArrToCompare == list3FirstArr[0,ind])
       rub, ind_data = tupleIndiceImage
       ind_data = np.array([ind_data])
       
       if len(ind_data[0]) == 1:
           singIndData.append(ind_data)
       else:
           print("Not singular")
           singIndData.append(ind_data[0,0])
           singIndData.append(ind_data[0,1])  
    
    newSingData = []
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))    
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd)     
          
    else:
        print("Not equal at phase 2")   
    
    list4FirstArr = np.array([np.array([list4[0]])[0]])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    singIndData = []
    
    for ind in range(0,len(list4FirstArr[0])):
       tupleIndiceImage = np.where(listArrToCompare == list4FirstArr[0,ind])
       rub, ind_data = tupleIndiceImage
       ind_data = np.array([ind_data])
       
       if len(ind_data[0]) == 1:
           singIndData.append(ind_data)
       else:
           print("Not singular")
           singIndData.append(ind_data[0,0])
           singIndData.append(ind_data[0,1])  
    
    newSingData = []
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))    
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd) 
            
    else:
        print("Not equal at phase 2")
    
    list5FirstArr = np.array([np.array([list5[0]])[0]])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    singIndData = []
    
    for ind in range(0,len(list5FirstArr[0])):
       tupleIndiceImage = np.where(listArrToCompare == list5FirstArr[0,ind])
       rub, ind_data = tupleIndiceImage
       ind_data = np.array([ind_data])
       
       if len(ind_data[0]) == 1:
           singIndData.append(ind_data)
       else:
           print("Not singular")
           singIndData.append(ind_data[0,0])
           singIndData.append(ind_data[0,1]) 
    
    newSingData = []
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))    
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
             
            indForFolderClustering.append(uniqueNewInd) 
            
    else:
        print("Not equal at phase 2")
        
    list6FirstArr = np.array([np.array([list6[0]])[0]])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    singIndData = []
    
    for ind in range(0,len(list6FirstArr[0])):
       tupleIndiceImage = np.where(listArrToCompare == list6FirstArr[0,ind])
       rub, ind_data = tupleIndiceImage
       ind_data = np.array([ind_data])
       
       if len(ind_data[0]) == 1:
           singIndData.append(ind_data)
       else:
           print("Not singular")
           singIndData.append(ind_data[0,0])
           singIndData.append(ind_data[0,1]) 
    
    newSingData = []
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))    
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
             
            indForFolderClustering.append(uniqueNewInd)        
           
    else:
        print("Not equal at phase 2")
        
    #%%
    
    
    ###############################################################################################################################################
    ###############################################################################################################################################
    ###############################################################################################################################################
    ###############################################################################################################################################
    ###############################################################################################################################################

    name_folder = first_clustering_storing_output + str(sequence_name) + "_7_8"
    newPath = os.path.join(parent_dir,name_folder)
    
    endInd = 8
    
    while True:
        if os.path.exists(newPath + '/'):
            endInd += 1
            if newPath[-2] == '_':
                newPathx = newPath[:-1]
                newPath = newPathx + str(endInd)
            elif newPath[-3] == '_':
                newPathx = newPath[:-2]
                newPath = newPathx + str(endInd)
        else:
            break
                
    os.mkdir(newPath)
     
    print("First Directory created")
    
    trainFolder = "Train_Results" 
    
    newPath = os.path.join(newPath + "/", trainFolder)   
    os.mkdir(newPath)
    
    newPath = newPath + "/"
    
    print("Second Directory created")
     
    sub_name_folder1 = "Class_1"
    newPath_1 = os.path.join(newPath + "/",sub_name_folder1)
    os.mkdir(newPath_1)
     
    sub_name_folder1 = "Class_2"
    newPath_2 = os.path.join(newPath + "/",sub_name_folder1)
    os.mkdir(newPath_2) 
         
    sub_name_folder1 = "Class_3"
    newPath_3 = os.path.join(newPath + "/",sub_name_folder1)
    os.mkdir(newPath_3)   
     
    sub_name_folder1 = "Class_4" 
    newPath_4 = os.path.join(newPath + "/",sub_name_folder1)
    os.mkdir(newPath_4)
     
    sub_name_folder1 = "Class_5"
    newPath_5 = os.path.join(newPath + "/",sub_name_folder1)
    os.mkdir(newPath_5) 
         
    sub_name_folder1 = "Class_6"
    newPath_6 = os.path.join(newPath + "/",sub_name_folder1) 
    os.mkdir(newPath_6)   
    
    counter_1 = 0
    counter_2 = 0 
    counter_3 = 0
    counter_4 = 0
    counter_5 = 0
    counter_6 = 0
    
    
    metricsIdTtrain1 = []
    metricsIdTtrain2 = []
    metricsIdTtrain3 = []
    metricsIdTtrain4 = []
    metricsIdTtrain5 = []
    metricsIdTtrain6 = []
    
    for indCluster, cluster in enumerate(indForFolderClustering):
        cluster_list = cluster.tolist()
        for ind_imageInCluster in cluster_list[0]:
    
            image_counter = cv2.imread(roi_bef + "/roi_image" + str(ind_imageInCluster) + ".jpg")
            image_counter_2 = cv2.imread(roi_after + "/roi_image" + str(ind_imageInCluster) + ".jpg")       
            
            if indCluster == 0:  
                cv2.imwrite(newPath_1 + "/" + "image_" + str(counter_1) + "_" + str(0) + ".jpg", image_counter) 
                cv2.imwrite(newPath_1 + "/" + "image_" + str(counter_1) + "_" + str(1) + ".jpg", image_counter_2)
                metricsIdTtrain1.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                counter_1 += 1
            if indCluster == 1:
                cv2.imwrite(newPath_2 + "/" + "image_" + str(counter_2) + "_" + str(0) + ".jpg", image_counter)
                cv2.imwrite(newPath_2 + "/" + "image_" + str(counter_2) + "_" + str(1) + ".jpg", image_counter_2)
                metricsIdTtrain2.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                counter_2 += 1
            if indCluster == 2: 
                cv2.imwrite(newPath_3 + "/" + "image_" + str(counter_3) + "_" + str(0) + ".jpg", image_counter)
                cv2.imwrite(newPath_3 + "/" + "image_" + str(counter_3) + "_" + str(1) + ".jpg", image_counter_2)
                metricsIdTtrain3.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                counter_3 += 1 
            if indCluster == 3:
                cv2.imwrite(newPath_4 + "/" + "image_" + str(counter_4) + "_" + str(0) + ".jpg", image_counter)
                cv2.imwrite(newPath_4 + "/" + "image_" + str(counter_4) + "_" + str(1) + ".jpg", image_counter_2)
                metricsIdTtrain4.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                counter_4 += 1
            if indCluster == 4:
                cv2.imwrite(newPath_5 + "/" + "image_" + str(counter_5) + "_" + str(0) + ".jpg", image_counter)
                cv2.imwrite(newPath_5 + "/" + "image_" + str(counter_5) + "_" + str(1) + ".jpg", image_counter_2)
                metricsIdTtrain5.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                counter_5 += 1
            if indCluster == 5:
                cv2.imwrite(newPath_6 + "/" + "image_" + str(counter_6) + "_" + str(0) + ".jpg", image_counter) 
                cv2.imwrite(newPath_6 + "/" + "image_" + str(counter_6) + "_" + str(1) + ".jpg", image_counter_2)
                metricsIdTtrain6.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                counter_6 += 1
    
    LenListsClusters1= [LenList_1, LenList_2, LenList_3, LenList_4, LenList_5, LenList_6]
    
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    
    m_1_f = []
    m_2_f = []
    m_3_f = []
    m_4_f = []
    m_5_f = []
    m_6_f = []
    
    
    metrics_afterClustering = []
    
    if len(metricsIdTtrain1) == 0: 
        print("Cluster not formed. Discarding this one")
    else:
        metricsIdTrain1 = np.array([metricsIdTtrain1]).T.tolist()
        metrics_afterClustering.append(metricsIdTrain1) 
        
        for m_1 in metricsIdTrain1:
            m_1_f.append(np.mean(np.array([m_1])[0,:,0]))    
        
    if len(metricsIdTtrain2) == 0: 
        print("Cluster not formed. Discarding this one")
    else:
        metricsIdTrain2 = np.array([metricsIdTtrain2]).T.tolist()
        metrics_afterClustering.append(metricsIdTrain2)
        
        for m_2 in metricsIdTrain2:
            m_2_f.append(np.mean(np.array([m_2])[0,:,0]))
        
    if len(metricsIdTtrain3) == 0: 
        print("Cluster not formed. Discarding this one")
    else:
        metricsIdTrain3 = np.array([metricsIdTtrain3]).T.tolist()
        metrics_afterClustering.append(metricsIdTrain3)
        
        for m_3 in metricsIdTrain3:
            m_3_f.append(np.mean(np.array([m_3])[0,:,0]))
        
    if len(metricsIdTtrain4) == 0: 
        print("Cluster not formed. Discarding this one")
    else:
        metricsIdTrain4 = np.array([metricsIdTtrain4]).T.tolist()
        metrics_afterClustering.append(metricsIdTrain4)
        
        for m_4 in metricsIdTrain4:
            m_4_f.append(np.mean(np.array([m_4])[0,:,0]))
        
    if len(metricsIdTtrain5) == 0: 
        print("Cluster not formed. Discarding this one")
    else:
        metricsIdTrain5 = np.array([metricsIdTtrain5]).T.tolist()
        metrics_afterClustering.append(metricsIdTrain5)
        
        for m_5 in metricsIdTrain5:
            m_5_f.append(np.mean(np.array([m_5])[0,:,0]))
        
    if len(metricsIdTtrain6) == 0: 
        print("Cluster not formed. Discarding this one")
    else:
        metricsIdTrain6 = np.array([metricsIdTtrain6]).T.tolist()
        metrics_afterClustering.append(metricsIdTrain6)
        
        for m_6 in metricsIdTrain6:
            m_6_f.append(np.mean(np.array([m_6])[0,:,0]))
            
            
    #################################################################################################################
    #################################################################################################################
    #%%
    
    labelsMetricsToScore = ['MSE', 'RMSE', 'RMSE_SINGLE', 'RMSE_SW', 'PSNR', 'UQI_SINGLE', 
                            'UQI', 'SSIMS', 'CSS', 'ERGAS', 'SCC',
                            'RASE', 'SAM', 'MSSSIM', 'VIFP', 'PSNRB',
                            'Mean', 'STD', 'Contrast', 'ASM', 'Max']
    
    stdListValues = []
    global_std = [] 
    
    secondMetricTable = np.array([metrics_afterClustering])[0].T.tolist() 
    
    listOfFlatten = []
    stdListValuesMetrics = []
    
    for indSec, secInd in enumerate(secondMetricTable):    
        flattenSecond = []
        flatten_list = [element for sublist in secInd for element in sublist]
        
        for fla in flatten_list:
            fla_one = np.array([fla])[0,0]
            flattenSecond.append(fla_one)
    
        listOfFlatten.append(flattenSecond)
    
    for indFlatten, metricFlatten in enumerate(listOfFlatten):
        stdValue = np.std(np.array([metricFlatten]))
        
        stdListValuesMetrics.append(stdValue)
    
    newstdlistValuesMetrics = []
    mainIndices = []
    
    sortedIndices = []
    listToPCA = []
    
    sorted_std_values = sorted(stdListValuesMetrics, reverse=True)
    
    for ind, stdValueSorted in enumerate(sorted_std_values):
        indStuff = np.where(np.array([stdListValuesMetrics]) == stdValueSorted)
        rub, indS = indStuff
        indS = np.array([indS])[0,0]
        indOfSorted = indS
        sortedIndices.append(indOfSorted)
    
    sortedIndicesToGo = sortedIndices[0:15]
    
    exlistToPCA = secRoundNewListMetricsArr.T.tolist()
    
    for ind_listPCA, listPCA in enumerate(exlistToPCA):
        if ind_listPCA in sortedIndicesToGo and ind_listPCA != 16: 
            listToPCA.append(listPCA) 
            
    filtered_metrics = []
    newstdlistValuesMetrics_sec = []
    metricsToPCA_analysis = [] 
    metricsToPCA_norm = []
    
    for ind_mat1, mat1 in enumerate(metrics_afterClustering):
        mat1_n = []
        for ind_mat2, mat2 in enumerate(mat1):
            mat2_n = [] 
            for ind_mat3, mat3 in enumerate(mat2):
                mat3 = mat3[0]
                mat2_n.append(mat3)
            mat1_n.append(mat2_n)  
        
        metricsToPCA_analysis.append(mat1_n)
     
    ## Standard normalization ########################################################################################################
    ##################################################################################################################################
    
    metricsToPCA_norm = []
    
    for met1 in listToPCA:     
        
            mean_value = np.mean(np.array([met1]))
            std_value = np.std(np.array([met1])) 
            metricPCA_norm1 = [] 
            
            for met2 in met1:
                metricPCA_norm1.append((met2-mean_value)/std_value)        
            
            metricsToPCA_norm.append(metricPCA_norm1)
    
    ##################################################################################################################################
    #%%
    
    dfxi = pd.DataFrame(data=metricsToPCA_norm)  
     
    pcai = PCA(n_components=None) 
    dfx_pcai = pcai.fit(dfxi)   
     
    X_pcai = pcai.transform(dfxi)   
    dfx_transi = pd.DataFrame(data=X_pcai)
    
    plt.scatter(dfx_transi[0], dfx_transi[1], c ="blue")
    plt.title("Correlation between first two PCA components")
    plt.xlabel("First PCA component")
    plt.ylabel("Second PCA component")
    plt.show()
    
    pca_coef_feat_first_comp = dfx_transi[0].tolist()
    abs_pca_coeff = np.array([abs(np.array([pca_coef_feat_first_comp])[0])])
    abs_pca_coeffList = abs_pca_coeff.tolist()
    
    ## sortedPCA_coeff = sorted(abs_pca_coeffList[0], reverse=True)
    sortedPCA_coeff = sorted(abs_pca_coeffList[0])
    
    ## ContrastAppending = sortedPCA_coeff[18] 
    
    sortedPCA_coeff = sortedPCA_coeff[0:4]
    ## sortedPCA_coeff = sortedPCA_coeff.append(ContrastAppending)
    
    sortedIndices2 = []
    
    for ind, pcaSorted in enumerate(sortedPCA_coeff):
        indStuff = np.where(np.array([abs_pca_coeffList]) == pcaSorted)
        rub, rub2, indS = indStuff  
        indOfSorted = indS 
        sortedIndices2.append(indOfSorted)    
         
    doubleSortedElements = []
    
    indicesSortedFromPCA = [] 
        
    for sorted_ind in sortedIndices2:
        if len(sorted_ind) == 2:
            doubleSortedElements.append(sorted_ind)
            list_aux = sorted_ind.tolist()
            if list_aux[0] in indicesSortedFromPCA:
                indicesSortedFromPCA.append(list_aux[1])
            else:
                indicesSortedFromPCA.append(list_aux[0])
        else:
            if len(sorted_ind) == 1:
                indicesSortedFromPCA.append(np.array([sorted_ind])[0,0])
                
    indicesSortedFromPCA.append(18)            
                
    remainingMetricsToClustering = [] 
    trainListData = []
    
    for ind in indicesSortedFromPCA: 
        if ind <= 21 and ind < len(exlistToPCA):
            remainingMetricsToClustering.append(labelsMetricsToScore[ind])   
            trainListData.append(exlistToPCA[ind]) 
             
    nclusters = 2
      
    trainListData = np.array([trainListData])[0].T.tolist()
    
    train_data = np.array([trainListData])[0].T
    treino_lenght = 370
    
    kmeans = KMeans(n_clusters=nclusters, max_iter=500, n_init=8).fit(trainListData) 
    Cluster_ID = kmeans.labels_ 
    centroides_A = kmeans.cluster_centers_   
    print("Centroides dos ",number_recommended_clusters, " clusters recomendados:\n", centroides_A)
    
    
    
    Cluster_ID_transpose = np.array([np.array([Cluster_ID]).T[0:treino_lenght,0]])
    
    objetos_c1 = []
    objetos_c2 = []
    ind_first = []
    ind_second = []
    
    for i in range (0, len(Cluster_ID_transpose[0])):
        if i<296:
            if Cluster_ID_transpose[0,i] == 0:
                objetos_c1.append(train_data[:,i])             
                i_1 = i
                ind_first.append(i)
            elif Cluster_ID_transpose[0,i] == 1:
                objetos_c2.append(train_data[:,i]) 
                i_2 = i
                ind_second.append(i)
                
    trainDataFurther = train_data.tolist()
    classClustering = []
    secFurther = []
    
    for tF in trainDataFurther:
        tF = tF[0:296]
        secFurther.append(tF)
    
    for indHere in range(0,296):
        if indHere in ind_first:
            classClustering.insert(indHere, 'A')        
        else:
            if indHere in ind_second:
                classClustering.insert(indHere, 'B')
    
    secFurther.insert(0, classClustering)
    classCl = secFurther[0]
    classAFurther = []
    classBFurther = []
    secFurther = np.array([secFurther])[0].T.tolist()
    
    for sInd, sF in enumerate(secFurther):
        if classCl[sInd] == 'A':
            classAFurther.append(sF[1:])
        else:
            if classCl[sInd] == 'B':
                classBFurther.append(sF[1:])
                
    clustering_inf_data = [classAFurther, classBFurther, nclusters, number_recommended_clusters, remainingMetricsToClustering]
    
    #%%
                
    dfxi1 = pd.DataFrame(data=classAFurther) 
    pcai1 = PCA(n_components=None) 
    dfx_pcai1 = pcai1.fit(dfxi1)   
     
    X_pcai1 = pcai1.transform(dfxi1)   
    dfx_transi1 = pd.DataFrame(data=X_pcai1)
    X_pcai1T = X_pcai1.T.tolist()
    
    dfxi2 = pd.DataFrame(data=classBFurther)  
    
    pcai2 = PCA(n_components=None) 
    dfx_pcai2 = pcai2.fit(dfxi2)    
    X_pcai2 = pcai2.transform(dfxi2)   
    dfx_transi2 = pd.DataFrame(data=X_pcai2)
    
    X_pcai2T = X_pcai2.T.tolist()
    
    centroid_pca_A = []
    centroid_pca_B = []
    
    for indA in range(0,5):
        if indA < len(X_pcai1T):
            mean_value = np.mean(np.array([X_pcai1T[indA]]))
            centroid_pca_A.append(mean_value)    
        
    for indB in range(0,5):
        if indB < len(X_pcai2T):
            mean_value = np.mean(np.array([X_pcai2T[indB]]))
            centroid_pca_B.append(mean_value)
    
    if len(centroid_pca_A) < len(centroid_pca_B):
        centroid_pca_B = centroid_pca_B[:len(centroid_pca_A)]
    elif len(centroid_pca_A) > len(centroid_pca_B):
        centroid_pca_A = centroid_pca_A[:len(centroid_pca_B)]
        
        
    distCentroidsPCA = np.linalg.norm(np.array([centroid_pca_A])[0]-np.array([centroid_pca_B])[0])
    
    #####
    thisDir = os.getcwd()
    dirResultsOutput = thisDir + '\\GraphsOutput\\'
    
    if os.path.isdir(dirResultsOutput) == False:  
    
        dirResults = os.path.join(dirResultsOutput) 
        os.mkdir(dirResults)    
    
    #####
    
    if len(dfx_transi1[0]) > 0 and len(dfx_transi1[1]) > 0:    
        plt1 = plt.scatter(dfx_transi1[0], dfx_transi1[1], c ="blue")
    if len(dfx_transi2[0]) > 0 and len(dfx_transi2[1]) > 0:
        plt2 = plt.scatter(dfx_transi2[0], dfx_transi2[1], c ="red")
        
    plt.legend((plt1, plt2),
               ('Class A', 'Class B'))
    plt.title("Correlation between first two PCA components")
    plt.xlabel("First PCA component")
    plt.ylabel("Second PCA component")    
    
    plt.savefig(dirResultsOutput + "pca_graph.png")
    
    plt.show() 
    
    data_results.append(clustering_inf_data)
    
    data_results.append(dirResultsOutput + "pca_graph.png")
    
    
    list1 = list(zip(*objetos_c1)) 
    list2 = list(zip(*objetos_c2)) 
    
    #%%
    
    list1ToDist = np.array([list1])[0].T
    list2ToDist = np.array([list2])[0].T 
    
    dists1 = []
    dists2 = []
    cent_1 = np.array([centroides_A[0,:]])
    cent_2 = np.array([centroides_A[1,:]])
    
    for i in range(0,len(list1ToDist)):
        
        a = list1ToDist[i,:]
        b = cent_1[0,:]    
        dist = np.linalg.norm(a-b)    
        dists1.append(dist) 
        
    for i in range(0,len(list2ToDist)):
        
        a = list2ToDist[i,:]
        b = cent_2[0,:]
        dist = np.linalg.norm(a-b)    
        dists2.append(dist)
    
    xDist1 = []
    xDist2 = []
    indD_1 = 0
    indD_2 = 0
    
    for ind in range(0, len(dists1)):
        indD_1 += 1
        xDist1.append(indD_1)
    
    for ind in range(0, len(dists2)):
        indD_2 += 1
        xDist2.append(indD_2)
        
    plt.scatter(xDist1, dists1, c ="blue")
    plt.title("Distance to centroid A")
    plt.xlabel("Number of point")
    plt.ylabel("Distance of points of first cluster to its centroid")
    plt.savefig(dirResultsOutput + "distances_firstCluster.png")
    plt.show()
    
    data_results.append(dirResultsOutput + "distances_firstCluster.png")
    
    plt.scatter(xDist2, dists2, c ="blue")
    plt.title("Distance to centroid B")
    plt.xlabel("Number of point")
    plt.ylabel("Distance of points of second cluster to its centroid")
    plt.savefig(dirResultsOutput + "distances_secondCluster.png")  
    plt.show() 
    
    data_results.append(dirResultsOutput + "distances_secondCluster.png")
    
        
    print(" -- Lists of clusters generated")
    
    for l in list1:
        LenList_1 = len(l)
    for l in list2:
        LenList_2 = len(l)
    
    #%%
        
    trainListData = np.array([trainListData])[0].T.tolist()   
        
    Y_euclidean = pdist(trainListData, metric='euclidean')
    Y_euclidean_square = squareform(Y_euclidean)
    Y_cityblock = pdist(trainListData, metric='cityblock')
    Y_euclidean_square = squareform(Y_cityblock)
    
    Z_euclidean_average = linkage(trainListData, method='average', metric='euclidean')
    Z_euclidean_ward = linkage(trainListData, method='ward', metric='euclidean')
    
    Z_cityblock_average = linkage(trainListData, method='average', metric='cityblock') 
    
    distances_from_euclidean_average = Z_euclidean_average[:,2].tolist()
    
    clustersList = Z_euclidean_average[:,0].tolist() + Z_euclidean_average[:,1].tolist()
    
    totNumbClusters_firstExp = np.max(np.array([clustersList])[0])
    
    numberObservationsForEachCluster_first = Z_euclidean_average[:,3].tolist()
    totObservationsFirst = np.sum(np.array([numberObservationsForEachCluster_first]))
    
    
    distances_from_euclidean_ward = Z_euclidean_ward[:,2].tolist()
    
    clustersList = Z_euclidean_ward[:,0].tolist() + Z_euclidean_average[:,1].tolist()
    
    totNumbClusters_secExp = np.max(np.array([clustersList])[0])
    
    numberObservationsForEachCluster_second = Z_euclidean_ward[:,3].tolist()
    totObservationsSec = np.sum(np.array([numberObservationsForEachCluster_second]))
    
    
    
    distances_from_cityblock_average = Z_cityblock_average[:,2].tolist()
    
    clustersList = Z_cityblock_average[:,0].tolist() + Z_euclidean_average[:,1].tolist()
    
    totNumbClusters_thirdExp = np.max(np.array([clustersList])[0])
    
    numberObservationsForEachCluster_third = Z_cityblock_average[:,3].tolist()
    totObservationsThird = np.sum(np.array([numberObservationsForEachCluster_third]))
    
    distancesMeanVar = [np.mean(np.array([distances_from_euclidean_average])), np.mean(np.array([distances_from_euclidean_ward])), np.mean(np.array([distances_from_cityblock_average]))]
    labelsMeasuresDistances = ['Euclidean Average', 'Euclidean Ward', 'Cityblock Average']
    
    maxMeanDistance = 0
    
    for indDist, distMeanValue in enumerate(distancesMeanVar):
        if distMeanValue > maxMeanDistance:
            maxMeanDistance = distMeanValue
            indMaxMeasureDistance = indDist 
    
    print("Selected Measure for distance between clusters: " + labelsMeasuresDistances[indMaxMeasureDistance])
    
    distancesBetClustersBestMeasure = []
    
    if indMaxMeasureDistance == 0:
        distancesBetClustersBestMeasure = Z_euclidean_average.tolist()
        numberClustersFromDist = totNumbClusters_firstExp
    else:
        if indMaxMeasureDistance == 1:
            distancesBetClustersBestMeasure = Z_euclidean_ward.tolist()
            numberClustersFromDist = totNumbClusters_secExp
        else:
            if indMaxMeasureDistance == 2:
                distancesBetClustersBestMeasure = Z_cityblock_average.tolist()
                numberClustersFromDist = totNumbClusters_thirdExp
                
    distance_output = []
                
    for dist1 in distancesBetClustersBestMeasure:
        distance_output2 = []
        for ind_dist2, dist2 in enumerate(dist1):
            if ind_dist2 != 2:
                distance_output2.append(int(dist2))
            else:
                distance_output2.append(dist2)
        distance_output.append(distance_output2)
        
    #### Comparison of number of clusters between distances approach and the clustering one:
    if  number_recommended_clusters == numberClustersFromDist:
        print("The above methods provide the same number of clusters")
    else:
        if number_recommended_clusters > numberClustersFromDist:
            print("The number of recommended clusters (from clustering approach) is higher than the number of clusters computed from dist-linkage method")
        else:
            if number_recommended_clusters < numberClustersFromDist:
                print("The number of clusters computed from dist-linkage method is higher than the number of recommended clusters (from clustering approach)")
    
    
    #%%     
    
    def convolve2D(image, kernel, padding=0, strides=1):
        # Cross Correlation
        kernel = np.flipud(np.fliplr(kernel))
    
        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[1]
    
        # Shape of Output Convolution
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros((xOutput, yOutput))
    
        # Apply Equal Padding to All Sides
        if padding != 0:
            imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
            imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
            print(imagePadded)
        else:
            imagePadded = image
    
        # Iterate through image
        for y in range(image.shape[1]):
            # Exit Convolution
            if y > image.shape[1] - yKernShape:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(image.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > image.shape[0] - xKernShape:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                    except:
                        break
    
        return output
    
    #%%
    
    k1 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    k2 = [[1, 0, -1],  [2, 0, -2], [1, 0, -1]]
    
    k1_arr = np.array([k1])[0]
    k2_arr = np.array([k2])[0]
    
    M_c = []
     
    #%%
    
    for video_image in range(0,count):           #### 321   ## from 361   ## to count = 637
        
            print("Analysing for image " + str(video_image) + " th")
        
            image = cv2.imread(IFVP + str(sequence_name) + "_2" + "/video_image%d.jpg" % video_image)        
            imx = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
            
           
            
            M1 = convolve2D(imx, k1)
            M2 = convolve2D(imx, k2)      
            
            
            print("Generating output for image " + str(video_image))
            
            comp_x = np.power(np.array([M1])[0], 2)      
            comp_y = np.power(np.array([M2])[0], 2) 
            
            sum_comps = comp_x + comp_y 
            
            gen_output = np.power(sum_comps, 1/2).astype(int)
            
            cv2.imwrite(roi_path + "/gen_output_%d.jpg" % video_image, gen_output)
            
            
            # M = [M1, M2]        
            
            # M_c.append(M)
            
            ################################################################################
           
            # elevation_map = cv2.imread(roi_path + "/elevation_map_%d.jpg" % video_image)       
            # elev_grey = cv2.cvtColor(elevation_map, cv2.COLOR_BGR2GRAY)  
            
    #%%      
    
    # for ind_m, m_x_c in enumerate(M_c):
        
    #     print("Generating output for image " + str(ind_m))
    #     m_x_1 = m_x_c[0]
    #     m_x_2 = m_x_c[1]
        
    #     comp_x = np.power(np.array([m_x_1])[0], 2)     
    #     comp_y = np.power(np.array([m_x_2])[0], 2) 
        
    #     sum_comps = comp_x + comp_y 
        
    #     gen_output = np.power(sum_comps, 1/2).astype(int)
        
    #     cv2.imwrite(roi_path + "/gen_output_%d.jpg" % ind_m, gen_output)
        
        
 
    
## Comment 
    #%% 
    
#     setCouplesToComparison = []
    
# ##  pathPythonFile = "C:/Research/SpeckleTraining/ffmpeg-5.0.1-full_build/bin"
    
#     for i in range(0,count):    
        
#         print("Joining for couple " + str(i))
        
#         imToCompare_1 = cv2.imread(pathPythonFile +  "/el_map_%d.png" % i)
#         imc1 = cv2.cvtColor(imToCompare_1, cv2.COLOR_BGR2GRAY)  
        
#         gen_output = cv2.imread(roi_path + "/gen_output_%d.jpg" % i)
#         imc2 = cv2.cvtColor(gen_output, cv2.COLOR_BGR2GRAY)  
         
#         coupleToComp = [imc1, imc2]
        
#         setCouplesToComparison.append(coupleToComp)    
     
#     #%% 
    
#     sizeImx = 1078
#     sizeImy = 1918
    
#     for ind_c, coup in enumerate(setCouplesToComparison):
#         im1_coup = coup[0]
#         im2_coup = coup[1]  
        
#         im1_coup = im1_coup[1:sizeImx+1, 1:sizeImy+1]
        
#         if len(im1_coup) == len(im2_coup) and len(im1_coup[0]) == len(im2_coup[0]):
#             imAbsDiff = np.zeros((len(im1_coup), len(im1_coup[0])))
            
#             for b in range(0,len(im1_coup[0])):
#                 for a in range(0, len(im1_coup)):
#                     imAbsDiff[a,b] = abs(im2_coup[a,b] - im1_coup[a,b])                
#             cv2.imwrite(roi_path + "/imAbsDiff_%d.jpg" % ind_c, imAbsDiff)     
#             print("Finding difference of images to achieve the best future k factor, for set " + str(ind_c))           
#         else:
#             print("Impossible to achieve comparison. Size of images not matched")
            
#     sums_all_pixels = []     
#     energyForAllImages = []   
    
#     for imi in range(0,count):
#         print("Reading image " + str(imi))
#         diffIm = cv2.imread(roi_path + "/imAbsDiff_%d.jpg" % imi)
#         diffImGrey = cv2.cvtColor(diffIm, cv2.COLOR_BGR2GRAY) 
        
#         sum_p = np.sum(diffImGrey)    
#         en_p = np.sum(np.power(diffImGrey,2)) 
        
#         sums_all_pixels.append(sum_p)    
#         energyForAllImages.append(en_p)
        
#     best_sum = np.min(np.array([sums_all_pixels])[0])
     
#     sizeImages = sizeImx*sizeImy
    
#     k_par = best_sum/sizeImages 
    
#     print("The best found value, for first approach, for k parameter was: " + str(k_par)) 
    
#     maxEnergy = np.max(np.array([energyForAllImages])[0])
#     meanEnergy = np.mean(np.array([energyForAllImages])[0])
#     minEnergy = np.min(np.array([energyForAllImages])[0])
     
#     k_par2 = best_sum/maxEnergy
#     k_par3 = best_sum/meanEnergy
#     k_par4 = best_sum/minEnergy
    
#     print("The best found value, for second approach, for k parameter was: " + str(k_par2))
#     print("The best found value, for third approach, for k parameter was: " + str(k_par3))
#     print("The best found value, for fourth approach, for k parameter was: " + str(k_par4))
    
#     ## Gap found for above defined K
#     percIntervalForK = [k_par2*100, k_par4*100, k_par3*100]
    
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
      
##    return percIntervalForK, executionTime, totCountImages
    return clustering_inf_data, executionTime, totCountImages 

