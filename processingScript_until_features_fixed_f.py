# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 00:05:15 2022

@author: marco
"""

from subVideoAnalysis import write_to_excel, find_min_images
from features_pan_auto_functions import features_auto
import pandas as pd

def write_number_tests_to_file(numberTests):
    with open('temp_numberTests.txt', 'a') as file:
        file.write(str(numberTests))
        
def write_to_excel_dirs_try(directory1, directory2, excel_file):
    try:
        # Try to read the existing Excel file
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty DataFrame
        df = pd.DataFrame(columns=["ROI's first group", "ROI's second group"])

    # Append the directories to the DataFrame
    new_row = {"ROI's first group": directory1, "ROI's second group": directory2}
    df = df.append(new_row, ignore_index=True)

    # Write the DataFrame to the Excel file
    df.to_excel(excel_file, index=False)

    print(f"Data written to '{excel_file}'")
        
def videoAnalysisMultipleVideosWithinOneJustFeatures(curTest, numberTests, tupleForProcessing, data_from_tests, direcPythonFile, filename_output_video, fps_out, limToCompare, buffer_imgs):
        
  #      numberTests += 1
       # if curTest == 0:
        # if True:
        #     data_from_tests = []
         
        # filename_output_videoN = [] 
        
        # for x in filename_output_video:
        #     x += '.mp4'
        #     filename_output_videoN.append(x)
        
        print("----------- \n\n -----------")
        print("Buffer Imgs: ")        
        print(buffer_imgs)
        print("----------- \n\n -----------")
        
        buffer_imgsx = buffer_imgs[curTest]
            
        filename_output_video += '.mp4'   
        
        limToCompare = 50
        
        new_namesF = [] 
        
        decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile = tupleForProcessing
  
        
        print("DecisorLevel: " + str(decisorLevel))
        print("MainPathVideoData: " + str(mainPathVideoData))
        print("SequenceName: " + str(sequence_name))
        print("DestinationPath: " + str(dest_path))
        print("MTSVideoPath: " + str(mtsVideoPath))
        print("MP4VideoFile: " + str(mp4VideoFile))
        print("IFVP: " + str(IFVP))
        print("LocationMP4File: " + str(locationMP4_file))
        print("ROIPath: " + str(roiPath))
        print("NewROIPath: " + str(newRoiPath))
        print("FirstClusteringStorageOutput: " + str(first_clustering_storing_output))
        print("PathPythonFile: " + str())
        
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
        
        if True:   
            
                import time
                startTime = time.time() 
            
                data_results = []
                
                ######################
                ######################
                ######################
                
                ## parent_dir = 'C:/Research/'
                parent_dir = dest_path    
                                                                
                infoI_seg = []
                infoJ_seg = []
                
                count = 0
                
                # pre_dir = mtsVideoPath  
                
                # #%%
                
                # ## IFVP = 'C:/Research/DataSequence_'
                # folder = IFVP + str(sequence_name) + '/'
                    
                # ##locationMP4_file = 'FilesFor_6_8_X1_'
                
                # name_folder = locationMP4_file + str(sequence_name)
                
                # pre_dirInit = "C:\\Users\\Other\\files_python\\py_scripts\\ffmpeg-5.0.1-full_build\\bin\\GUI\\"
                 
                # path_in = pre_dirInit + "test_" + "00" + str(sequence_name) + ".avi"    ## .mts 
                
                # mp4_initDir = os.path.join(mp4VideoFile + name_folder + "\\") 
                
                # new_mp = ""
                
                # if " " in mp4_initDir or "\t" in mp4_initDir or "\n" in mp4_initDir or "\r" in mp4_initDir:
                
                #     for x in mp4_initDir:
                #         if x != " " and x != "\t" and x != "\n" and x != "\r":
                #             new_mp += x
                #     mp4_initDir = new_mp
                    
                #     mp4_initDir = mp4_initDir[1:]
                    
                #     mps = mp4_initDir.split('\\')
                #     addFile = mps[-2]
                #     mx = addFile.split(':')
                    
                #     mx_n = ""
                    
                #     for m in mx:
                #         if len(m) != 0:
                #             mx_n += m            
                     
                #     mps[-2] = mx_n
                    
                #     mp4_initDir_new = ""
                    
                #     for x in mps[:-1]:
                    
                #         mp4_initDir_new += x + '\\'
                        
                #     mp4_initDir = mp4_initDir_new
                
                # i = 1
            
                # while True:
                #     if os.path.exists(mp4_initDir):
                #         mp4_initDir_p = mp4_initDir[:-1]
                #         mp4_initDir_p_x = mp4_initDir_p[:-1]
                #         mp4_initDir_p = mp4_initDir_p_x + str(i)
                #         mp4_initDir = mp4_initDir_p + "\\"
                        
                #         i += 1
                #     else:
                #         break
                
                
                # os.mkdir(mp4_initDir) 
                
                # seqnew = ""
                
                # for seqn in str(sequence_name):
                #     if seqn != " " and seqn != "\r" and seqn != "\n" and seqn != "\t" and seqn != ":":
                #         seqnew += seqn
                        
                # sequence_name_str = seqnew
                # sequence_name = int(sequence_name_str) 
                
                # path_out = mp4_initDir + "test_" + "00" + str(sequence_name) + '.mp4'
                # path_out_mp4 = pre_dirInit + "test_" + "00" + str(sequence_name) + ".mp4"    ## .mts     
            
                
                # def pairedNumber(n1, div):
                     
                #     if not (n1%2 == 0): 
                #         a = int(n1/2)
                         
                #         if n1 > a*2: 
                #             n2 = a*2
                #         else: 
                #             n2 = n1       
                        
                #         return n2 
                #     return n1    
                 
                
                # def loadMP4_file(path_in, path_out):
                #             name_in = "test_" + "00" + str(sequence_name) + ".avi"
                #             name_out = "test_" + "00" + str(sequence_name) + ".mp4"
                #             cmd = 'ffmpeg -i  ""C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/' + name_in + "" + ' "" C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/' + name_out + ""
                #             print("ffmpeg command: " + cmd)
                            
                #             cmd_arr = shlex.split(cmd)
                #             cmd_arr = np.array([cmd_arr])
                #             cmd_arr = np.delete(cmd_arr, 3)
                #             cmd = np.array([cmd_arr])[0].tolist() 
                            
                #             subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=False)
                #             dir_out = path_out            
                #             time.sleep(5) 
                #             return dir_out 
                
                # if os.path.isdir(pre_dir+name_folder): 
                #     print("Directory already exists !!!")
                # else: 
                
                #     newPath = os.path.join(pre_dir, name_folder)   
                    
                #     new_path_n = ""
                    
                #     for n in newPath:
                #         if n != " " and n != "\t" and n != "\r" and n != "\n" and n != "\\":
                #             new_path_n += n
                    
                #     new_path_n = new_path_n[1:]
                #     new_path_s = new_path_n.split('/')
                    
                #     sp_n = ""
                    
                #     for ind_sp, sp in enumerate(new_path_s):
                #         if ind_sp < len(new_path_s)-1:
                #             sp_n += sp + '/'
                    
                #     rem = new_path_s[-1]
                    
                #     remx = rem[1:-3] + rem[-1]
                #     sp_n += remx
                #     newPath = sp_n 
                    
                #     j = -1
                #     cod = 0
                    
                #     while True:
                #         if '/' in newPath:
                #             cod = 1
                #         elif "\\" in newPath:
                #             cod = 2
                        
                #         if cod == 1:
                #             if os.path.exists(newPath + '/'):
                #                 j += 1
                                
                #                 if newPath[-2] == '_':
                #                     newPath = newPath[:-2] 
                #                 if newPath[-3] == '_':
                #                     newPath = newPath[:-3] 
                                    
                #                 newPath = newPath + "_" + str(j)
                #                 print("New Path: "+ newPath)
                #             else:
                #                 break  
                #         elif cod == 2:
                #             if os.path.exists(newPath + "\\"):
                #                 j += 1
                                
                #                 if newPath[-2] == '_':
                #                     newPath = newPath[:-2] 
                #                 if newPath[-3] == '_':
                #                     newPath = newPath[:-3] 
                                    
                #                 newPath = newPath + "_" + str(j)
                #                 print("New Path: "+ newPath)
                #             else:
                                
                #                 break            
                    
                #     if newPath[0] != 'C':
                #         newPath = "C" + newPath
                        
                #     os.mkdir(newPath)  
                    
                #     print("Directory created")
                 
                     
                #     if os.path.isfile('path_out' + '/' + str(sequence_name) + '.mp4'):
                #         print("MP4 File already exists inside directory") 
                #     else:
                #         print(path_in)
                #         print(path_out)
                #         src_dir = loadMP4_file(path_in, path_out_mp4) 
                #         print("path_out_mp4: \t" + path_out_mp4)
                #         print("path_out: \t" + path_out)
                #         time.sleep(100)            
                #         os.rename(path_out_mp4, path_out) 
                #         while(os.path.exists(path_out) == False): 
                #             time.sleep(5)
                #         print("MP4 file loaded")          
                    
                #     time.sleep(100)    
                    
                #     src_dir = path_out 
                
                
                
       #          src_dir = direcPythonFile + filename_output_video  
                
       #   ##       print("Source Directory for MP4 file: " + src_dir)
                
       #          vidcap = cv2.VideoCapture(filename_output_video) 
               
                
       #          file_stats = os.stat(filename_output_video) 
       #          print(str(file_stats.st_size))
                
       #          if int(file_stats.st_size) < 1000:  ## Size of sub-video file, in Bytes (258 Bytes - standard fail)
       #              print("Error reading video file. \n Skipping this one")
       #    ##      else:
       #          if True:
                
       #   #           fps_out = 50    
       #              index_in = -1
       #              index_out = -1    
       #              reader = imageio.get_reader(filename_output_video)
       #              fps_in = reader.get_meta_data()['fps'] 
       #              print("fps in: " + str(fps_in))
       #   #           limToCompare = 50
                    
                   
                    
       #              IFVP_n = ""
                    
       #              for x in IFVP:
       #                  if x != "\n" and x != "\r" and x != "\t" and x != " ":
       #                      IFVP_n += x
                            
       #              IFVP = IFVP_n  
                    
       #              newPath = os.path.join(IFVP + str(sequence_name) + "_2")
                   
       #  ##            print("New Path: " + str(newPath))
       #   ##           print(os.path.exists(newPath + '/'))
                    
                     
       #    ##          print(vidcap.read())
       # ##             print(limToCompare)
                    
       #              while True: 
                        
       #                  success, image = vidcap.read()
       #                  if success and count < limToCompare:
                            
       #               ##       print("\n\n Here on success !!! \n\n")
                            
       #                      if os.path.exists(newPath + '/'):
       #                          print("Directory already exists !!!")
                            
       #                      else:                      
       #                          newPath = os.path.join(IFVP + str(sequence_name) + "_2")
       #                          pathBig = IFVP + str(sequence_name) + "_2"
       #                          pathb = ""
       #                          if "\n" in pathBig or "\r" in pathBig or "\t" in pathBig or " " in pathBig:
       #          ##                    print("PathBig")
       #                              for x in pathBig:
       #                                  if x != "\n" and x != "\r" and x != "\t" and x != " ":
       #                                    pathb += x
       #                              if pathb[0] == ':':
       #                                  pathb = pathb[1:]                                 
                                    
       #                              newPath = pathb 
                                 
       #                          if newPath[0] == ':':
       #                              newPath = newPath[1:]
                                    
       #                          l = -1
                                
       #                          while True:
       #                              if os.path.exists(newPath + '/') or os.path.exists(newPath + "\\"):
       #                                  l += 1
       #                                  if l > 0:
       #                                      nps = newPath.split('/')
       #                                      x = nps[-1]
       #                                      xs = x.split('_')
       #                                      xs = xs[:-1]
       #                                      xns = "DataSequence__"
                                            
       #                                      for ind_ps, xps in enumerate(xs):
       #                                          if ind_ps >= 2:
       #                                              xns += xps + '_'
       #                                      addrem = xns[:-1]
       #                                      remd = ""                                    
       #                                      for n in nps:
       #                                          remd += n + '/'
                                            
       #                                      newPath = remd + addrem
                                            
       #                                  newPath = newPath + "_" + str(l) 
       #           ##                       print("Here for path " + str(l))
       #                              else:
       #                                  break  
                                 
       #                          os.mkdir(newPath)      
       #          ##                print("Another directory created")
                             
       #                      cv2.imwrite(newPath + "/video_image%d.jpg" % count, image)     
       #                      count += 1
       #                      print("count: " + str(count))
       #                  else:  
       #                      break           
                
       #           ##   print("Images loaded") 
        if True:        
    #             totCountImages = count 
    #     ##        print("totCountImages: " + str(totCountImages))
                    
    #             #%%
                
                roi_path = os.path.join(mainPathVideoData + roiPath + str(sequence_name) + '/')
                
                rpx = 0
                 
                i = 0
                
                while True:
                
                    if os.path.exists(roi_path):            
                    
                        if '/' in roi_path:
                            rpx = 1
                        elif "\\" in roi_path:
                            rpx = 2
                            
                        roi_path_p = roi_path[:-1]
                        roi_path_p_x = roi_path_p[:-1]
                        roi_path_p = roi_path_p_x + str(i)
                        
                        if rpx == 1:
                            roi_path = roi_path_p + '/'
                        elif rpx == 2:
                            roi_path = roi_path_p + "\\"
                       
                        i += 1
                    else:
                        break  
                
                roi_path_clean = ""
                roi_path_clean_2 = ""
                for r in roi_path:
                    if r != " " and r != "\n" and r != "\t" and r != "\r":
                        roi_path_clean += r
                
                roi_path_clean = roi_path_clean[1:]
            
                for ind_r, r in enumerate(roi_path_clean):
                    if not(ind_r != 1 and r == ':'):
                        roi_path_clean_2 += r
                    
                roi_path = roi_path_clean_2
                
                e = -1
                
                while True:
                
                    if os.path.exists(roi_path + '/'):
                        e += 1
                        roi_pathS = roi_path.split('/')
                        remRoi = roi_pathS[:-1]
                        remRoiS = ""
                        
                        for rm in remRoi:
                            remRoiS += rm + '/'
                        
                        nameS = roi_pathS[-1]
                        nameS_sp = nameS.split('_')
                        newNameS = ""
                        
                        for indna, na in enumerate(nameS_sp):
                            if indna < 3:
                                newNameS += na + '_'
                        newNameS += str(e)
                        roi_path = remRoiS + newNameS
                    else:
                        break
                
                if roi_path[0] == '/':
                    roi_path = 'C:' + roi_path 
                
                lastF = 0
                for ind_f, f in enumerate(roi_path):
                    if f == '_':
                        lastF = ind_f
                
                for ind in range(0,500):
                
                    roi_path = roi_path[:lastF] + str(ind) + '/'
                    
                    if not os.path.exists(roi_path):
                
                        os.mkdir(roi_path)
                        break
                    
                
                
    ##            os.mkdir(roi_path)
    
                print("Count: " + str(count))
              
                dec_thresh = decisorLevel  
                
                if True:
                
           ##      for video_image in range(0,count):
               
                   totCountImages = 0
                   
                   if len(buffer_imgsx) > 0:
               
                       for video_image, img in enumerate(buffer_imgsx):
                        
                #             print("Analysing for image " + str(video_image) + " th")
                             
                             ## newPath + "/video_image%d.jpg
                        
                             image = img   
                             totCountImages += 1
                             print(image) 
                            
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
                            
                             decThreshrem = "" 
                            
                             print("Type of dec_thresh: " + str(dec_thresh))
                            
                       #      if type(dec_thresh) != 'int':
                             if isinstance(dec_thresh, (int,float)):
                                 dec_thresh_int = dec_thresh
                             elif isinstance(dec_thresh, str):
                                 for t in dec_thresh: 
                                     if t != '\r' and t != '\t' and t != '\n' and t != ':' and t != ' ':
                                         decThreshrem += t
                                 dec_thresh = decThreshrem
                                 
                                 dec_thresh_int = int(dec_thresh)
                             # else: 
                   #          dec_thresh_int = dec_thresh
                            
                             for j in range(0,len(imageOverlay[0])):
                                 for i in range(0,len(imageOverlay)):
                                     if imageOverlay[i,j] >= dec_thresh_int:         ### ------------------------
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
                   else:
                       print("Buffer empty. Error detected ...")
                   
                   if True:
                
                     count = 0
                    
                     fixed_roi_path = os.path.join(mainPathVideoData + newRoiPath + str(sequence_name) + '/')   
                    
                     rpx = 0
                    
                     i = 0
                    
                     fixed_new = ""
                     fixed_new_2 = ""
                    
                     for f in fixed_roi_path:
                         if f != "\t" and f != "\n" and f != "\r" and f != " ":
                             fixed_new += f
                     fixed_new = fixed_new[1:]
                    
                     for ind_r, r in enumerate(fixed_new):
                         if not(ind_r != 1 and r == ':'):
                             fixed_new_2 += r
                    
                     fixed_roi_path = fixed_new_2 
                    
                     while True:
                    
                         if os.path.exists(fixed_roi_path):            
                        
                             if '/' in fixed_roi_path:
                                 rpx = 1
                             elif "\\" in fixed_roi_path:
                                 rpx = 2
                                
                             fixed_roi_path_p = fixed_roi_path[:-1]
                             fixed_roi_path_p_x = fixed_roi_path_p[:-1]
                             fixed_roi_path_p = fixed_roi_path_p_x + str(i)
                            
                             if rpx == 1:
                                 fixed_roi_path = fixed_roi_path_p + '/'
                             elif rpx == 2:
                                 fixed_roi_path = fixed_roi_path_p + "\\"
                           
                             i += 1
                         else: 
                             break       
                    
                     if fixed_roi_path[0] == '/':
                         fixed_roi_path = 'C:' + fixed_roi_path 
                    
                     lastF = 0
                     for ind_f, f in enumerate(fixed_roi_path):
                         if f == '_':
                             lastF = ind_f
                    
                     for ind in range(0,500):
                    
                         fixed_roi_path = fixed_roi_path[:lastF] + str(ind) + '/'
                        
                         if not os.path.exists(fixed_roi_path):
                    
                             os.mkdir(fixed_roi_path)
                             break
                    
                     ## fixed_roi_path = 'C:/Research/Approach317_new' 
                    
                     print("totCountImages: " + str(totCountImages))
                    
                     print("\n\n\n ---------- \n\n Fixed ROI Path before: \n\n\n ---------- \n\n")
                     print(fixed_roi_path) 
                    
                     if totCountImages > 0:
                         write_to_excel('images_singleVideo.xlsx',curTest,totCountImages)
                     else:
                   #      numberTests-= 1
                         write_number_tests_to_file(numberTests)
                         
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
                            
                            
                            
                            
                     from getCurrentDateAndTime import getDateTimeStrMarker          
                     dateTimeMarker = getDateTimeStrMarker() 
                    
                     print("DecisorLevel: " + str(decisorLevel))
                     print("MainPathVideoData: " + str(mainPathVideoData))
                     print("SequenceName: " + str(sequence_name))
                     print("DestinationPath: " + str(dest_path))
                     print("MTSVideoPath: " + str(mtsVideoPath))
                     print("MP4VideoFile: " + str(mp4VideoFile))
                     print("IFVP: " + str(IFVP))
                     print("LocationMP4File: " + str(locationMP4_file))
                     print("ROIPath: " + str(roiPath))
                     print("NewROIPath: " + str(newRoiPath))
                     print("FirstClusteringStorageOutput: " + str(first_clustering_storing_output))
                     print("PathPythonFile: " + str(pathPythonFile))
                     
                     data_from_tests.append((decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile, fixed_roi_path, dateTimeMarker))
                    
                     print("Here on test ...")
                    
                     print("Current test: " + str(curTest))
                     print("Total number of tests: " + str(numberTests))
               
        if curTest < numberTests-1:
               print("Returning this: ")   ## 0
               for d in data_from_tests:
                  for x in d: 
                       print(x) 
                         
               return data_from_tests  
        elif curTest == numberTests-1:  
            
                # if curTest == numberTests:
                #     curTest -= 1
         
                #%%
            
                ## - repeat the code above for all tests and compare iterativally
                ## if it's the last one, procceed. If not, save the above variables and go for the next test 
                
                print("Length for data from tests: " + str(len(data_from_tests)))    
                
                print("------- \n\n\n Arrived here \n\n\n ------ ")
                clustering_output = []
                 
                M = 0
                
                metricsD = []
                 
                if len(data_from_tests) > 0:
                    
                    print("Got 1")
                    
                    if  curTest == numberTests-1:   ## for
                        
                        # if curTest == numberTests:
                        #     curTest -= 1
                             
                        couples_combs = []
                        ## numberTests = 10
                        
                        not_include = False 
                        
                        print("Got 2")
                    
                        for x in range(0, numberTests):
                            for y in range(0, numberTests):
                                M += 1
                                if x != y:
                                    
                                    couple_number_tests = (x,y)           
                                    
                                    if len(couples_combs) > 0:
                                        for x in range(0, numberTests):
                                            for y in range(0, numberTests):
                                                couple_x = (x,y)
                                                if couple_number_tests[0] == couple_x[1] and couple_number_tests[1] == couple_x[0]:
                                                    not_include = True
                                    
                                    if not not_include: 
                                        
                                        totCountImages = find_min_images('images_singleVideo.xlsx')
                                        couples_combs.append(couple_number_tests)
                                #         proc_algorithm(x, y)
                                        data_test_one = data_from_tests[x]
                                        data_test_two = data_from_tests[y]
                                        
                                        dest_path_one = data_test_one[3] 
                                        dest_path_two = data_test_two[3]
                                        
                                        sequence_name_one = data_test_one[2]
                                        sequence_name_two = data_test_two[2]
                                        
                                        dateTimeMarker_one = data_test_one[-1]
                                        dateTimeMarker_two = data_test_two[-1]
                                        
                                        fixed_roi_path_one = data_test_one[-2]
                                        fixed_roi_path_two = data_test_two[-2]
                                        
                                        if not os.path.exists(fixed_roi_path_one):                                        
                                            os.mkdir(os.path.join(fixed_roi_path_one))
                                            
                                        if not os.path.exists(fixed_roi_path_two):                                            
                                            os.mkdir(os.path.join(fixed_roi_path_two))
                                        
                                        print("fixed_roi_path_one: " + str(fixed_roi_path_one))
                                        print("fixed_roi_path_two: " + str(fixed_roi_path_two))
                                        
                                        
                                        #######################################
                                        #######################################
                                        #######################################
                                        #######################################
                                        #######################################
                                        
                                        pathRoiStart = dest_path_one + 'modRoisFirstVideo_' + dateTimeMarker_one
                                        pathRoiEnd = dest_path_two + 'modRoisSecVideo_' + dateTimeMarker_two
                                    
                                        roi_bef = pathRoiStart + str(sequence_name_one)
                                        roi_after = pathRoiEnd + str(sequence_name_two)     
                                         
                                        #%%
                                        
                                        # start_mark_before_inter = 0
                                        # end_mark_before_inter = 50
                                         
                                        # numberImagesBef = end_mark_before_inter - start_mark_before_inter + 1 
                                         
                                        # start_mark_after_inter = count-50
                                        # end_mark_after_inter = count
                                         
                                        # numberImagesAfter = end_mark_after_inter - start_mark_after_inter + 1
                                         
                                        newPathA = os.path.join(roi_bef)    
                                        
                                        new_newPathA = ""
                                        
                                        for p in newPathA:
                                            if p != '\n' and p != "\n" and p!= "\t" and p != " ":
                                                new_newPathA += p
                                                
                                        if new_newPathA[1] == 'C':
                                            new_newPathA = new_newPathA[1:]
                                        
                                        if new_newPathA[0] != 'C':
                                            new_newPathA = 'C' + new_newPathA
                                        
                                        newPathA = new_newPathA 
                                        
                                        itA = 0
                                        
                                        if '/' in newPathA:
                                        
                                            while os.path.exists(newPathA + '/'):
                                                
                                                if newPathA[-2] == '_':
                                                    newPathA = newPathA[:-2] 
                                                if newPathA[-3] == '_':
                                                    newPathA = newPathA[:-3] 
                                                    
                                                newPathA += '_' + str(itA)
                                                itA += 1 
                                            else:    
                                                os.mkdir(newPathA)
                                        elif "\\" in newPathA:
                                            
                                            if newPathA[-2] == '_':
                                                newPathA = newPathA[:-2] 
                                            if newPathA[-3] == '_':
                                                newPathA = newPathA[:-3] 
                                        
                                            while os.path.exists(newPathA + "\\"):
                                                newPathA += '_' + str(itA)
                                                itA += 1 
                                            else:    
                                                os.mkdir(newPathA)
                                         
                                        newPathB = os.path.join(roi_after)   
                                        
                                        new_newPathB = ""
                                        
                                        for p in newPathB:
                                            if p != '\n' and p != "\n" and p!= "\t" and p != " ":
                                                new_newPathB += p
                                                
                                        if new_newPathB[1] == 'C':
                                            new_newPathB = new_newPathB[1:]
                                        
                                        if new_newPathB[0] != 'C':
                                            new_newPathB = 'C' + new_newPathB
                                        
                                        newPathB = new_newPathB
                                        
                                        itB = 0 
                                        
                                        if '/' in newPathB:
                                        
                                            while os.path.exists(newPathB + '/'):
                                                
                                                if newPathB[-2] == '_':
                                                    newPathB = newPathB[:-2] 
                                                if newPathB[-3] == '_':
                                                    newPathB = newPathB[:-3] 
                                                    
                                                newPathB += '_' + str(itB)
                                                itB += 1 
                                            else:    
                                                os.mkdir(newPathB)
                                        elif "\\" in newPathB:
                                        
                                            while os.path.exists(newPathB + "\\"):
                                                
                                                if newPathB[-2] == '_':
                                                    newPathB = newPathB[:-2] 
                                                if newPathB[-3] == '_':
                                                    newPathB = newPathB[:-3] 
                                                    
                                                newPathB += '_' + str(itB)
                                                itB += 1 
                                            else:    
                                                os.mkdir(newPathB)
                                        
                                ##         os.mkdir(newPathB)  
                                          
                                        ind_bef = 0
                                        ind_after = 0
                                        
                                        roiPathBefA = ""
                                        
                                        for p in roi_bef:
                                            if p != '\n' and p != "\n" and p!= "\t" and p != " ":
                                                roiPathBefA += p
                                        
                                        roi_bef = roiPathBefA
                                        roi_bef = roi_bef[1:]
                                        
                                        roiPathAfterB = ""
                                        
                                        for p in roi_bef:
                                            if p != '\n' and p != "\n" and p!= "\t" and p != " ":
                                                roiPathAfterB += p
                                        
                                        roi_after = roiPathAfterB
                                        roi_after = roi_after[1:]
                                        
                                        print("Complete ROI Bef: " + roi_bef)
                                        print("Complete ROI After: " + roi_after)   
                                        
                                        count = totCountImages
                                        
                                        print("Fixed ROI path: " + fixed_roi_path)
                                         
                                        for i in range(0, count):
                                            imx = cv2.imread(fixed_roi_path + "/roi_image" + str(i) + ".jpg")
                                              ## fixed_roi_path_one
                                            print("Imx: ")
                                            print(imx)
                                            
                                            imxa = cv2.cvtColor(imx, cv2.COLOR_BGR2GRAY) 
                                            print(roi_bef + "/roi_image")
                                            print("Image before written")
                                            
                                            if roi_bef[0] != 'C':
                                                roi_bef = 'C' + roi_bef 
                                            
                                            
                                            if '/' in roi_bef:
                                            
                                                if not os.path.exists(roi_bef + '/'):
                                                    print("Creating new path for left ROI image")
                                                    os.mkdir(roi_bef) 
                                                else:
                                                    print("Path from left already exists. Overwritting ...")
                                                    
                                            elif "\\" in roi_bef:
                                                if not os.path.exists(roi_bef + "\\"):
                                                    print("Creating new path for left ROI image")
                                                    os.mkdir(roi_bef) 
                                                else:
                                                    print("Path from left already exists. Overwritting ...")
                                                                                       
                                                
                                            if not os.path.exists(fixed_roi_path_two):
                                                print("Creating new path for right ROI image")
                                                os.mkdir(fixed_roi_path_two) 
                                            else:
                                                print("Path from right already exists. Overwritting ...")
                                            
                                            
                                              
                                            cv2.imwrite(roi_bef + "/roi_image%d.jpg" % ind_bef, imxa)
                                            
                                            ind_bef += 1
                                            
                                            
                                            roi_after = roi_after[1:]
                                            
                                            print("ROI after now:  " + str(roi_after))
                                            
                                            indUafter = 0
                                            
                                            for ind_r, r in enumerate(roi_after):
                                                if r == 'U':
                                                    indUafter = ind_r
                                                    break
                                                    
                                            indUafter = int(indUafter)
                                            
                                            print("indUafter: " + str(indUafter))
                                                
                                            if indUafter != 0:
                                                roi_afterX = 'C:/' + roi_after[indUafter:] 
                                            else:
                                                roi_afterX = 'C:/' + roi_after[:] 
                                                 
                                            roi_after = roi_afterX
                                            
                                            if roi_after[0] != 'C':
                                                roi_after = 'C' + roi_after
                                             
                                            
                                            if '/' in roi_after:
                                                if roi_after[0] != 'C' and roi_after[1] != ':' and roi_after[2] != '/':
                                                    indU = 0
                                                    ok = False
                                                    for ind_r, r in enumerate(roi_after):
                                                        if r == 'U':
                                                            indU = ind_r
                                                            ok = True
                                                            break
                                                    if ok:
                                                        roi_after = roi_after[indU:]
                                                    
                                                    print("This yes: ")
                                                    print(roi_after)
                                                    roi_after = 'C:/' + roi_after 
                                                if not os.path.exists(roi_after + '/'):
                                                    print("Creating new path for right ROI image")
                                                    os.mkdir(roi_after)  
                                                else:
                                                    print("Path from left already exists. Overwritting ...")
                                                    
                                            elif "\\" in roi_after: 
                                                if roi_after[0] != "C" and roi_after[1] != ":" and roi_after[2] != "\\":
                                                    indU = 0
                                                    ok = False
                                                    for ind_r, r in enumerate(roi_after):
                                                        if r == 'U':
                                                            indU = ind_r
                                                            ok = True
                                                    if ok:
                                                        roi_after = roi_after[indU:]
                                                        
                                                        
                                                    roi_after = "C:\\" + roi_after
                                                if not os.path.exists(roi_after + "\\"):
                                                    print("Creating new path for right ROI image")
                                                    os.mkdir(roi_after) 
                                                else:
                                                    print("Path from right already exists. Overwritting ...")
                                                     
                                            
                                            imx = cv2.imread(fixed_roi_path_two + "/roi_image" + str(i) + ".jpg")
                                            print(fixed_roi_path_two + "/roi_image")
                                            print("Image after written")
                                     
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
                                        
                                        print("Tot count images: " + str(totCountImages))
                                        
                                        for ind in range(0,int(totCountImages/2)):
                                            im1 = cv2.imread(roi_bef + "/roi_image%d.jpg" % ind)
                                            im2 = cv2.imread(roi_after + "/roi_image%d.jpg" % ind)   ##
                                            
                                            print("----- \n Image 1: \n")
                                            print(im1)
                                            
                                            print("----- \n Image 2: \n")
                                            print(im2) 
                                            
                                            ori_img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
                                            ori_img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)  
                                           
                                            couple_images = [ori_img1, ori_img2]               
                                                   
                                            bigCoupleImages.append(couple_images)
                                            couple_images = []
                                    
                                        print("Length for bigCoupleImages: " + str(len(bigCoupleImages))) 
                                        
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
                                            
                                            print("Metrics values: \n")
                                            for ind_newM, newM in enumerate(newMetricsValues):    
                                                metrics_values.append(newM)
                                                print(str(ind_newM) + " - " + str(newM))
                                               
                                            
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
                                    
                                        print("Length for setMetricsValues: " + str(len(setMetricsValues)))
                                            
                                        newListMetrics = []
                                        
                                        for ind,  metricValue in enumerate(setMetricsValues):
                                            number_No_zeros = 0
                                            number_No_zeros = np.count_nonzero(np.array([metricValue]))
                                            
                                            print("Number no zeros: " + str(number_No_zeros))
                                            if int(number_No_zeros) > 0:     ## int(number_No_zeros) > 12
                                                newListMetrics.append(metricValue) 
                                    
                                        print("Length for newListMetrics: " + str(len(newListMetrics)))
                                        
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
                                        
                                        time.sleep(5)
                                        
                                        newf = features_auto(labelsFeatQualityValues)
                                         
                                        print("New created functions inside extra.py file: ")
                                        print(newf)
                                           
                                        print("Length for listMetricsForEvaluation: " + str(len(listMetricsForEvaluation)))
                                        print("Length for each one: ")
                                        if len(listMetricsForEvaluation) > 0:
                                            for l in listMetricsForEvaluation:
                                                print(" - " + str(len(l)))
                                            
                                            
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
                                                
                                                print("Preliminar metric eval.: " + str(len(newMetricEval)))
                                                print("Number no zeros: " + str(number_No_zeros))
                                                
                                                if number_No_zeros > 0:   ## number_No_zeros > len(newMetricEval)/2
                                                    secRoundNewListMetrics.append(newMetricEval)
                                                else: 
                                                    metricsIndicesDeleted.append(ind_new_sec)
                                        
                                        for ind_d, d in enumerate(labelsFeatQualityValues):
                                            if ind_d not in metricsIndicesDeleted: 
                                                new_namesF.append(d)
                                        
                                        
                                        test_size = 0.2
                                        newsecRoundNewListMetrics = []
                                         
                                        lenMax = 0
                                        
                                        print("Length of secRoundNewListMetrics: " + str(len(secRoundNewListMetrics)))
                                        
                                        if len(secRoundNewListMetrics) > 0:
                                        
                                            for metricSec in secRoundNewListMetrics:
                                                lenMax = len(metricSec)   
                                                 
                                            if lenMax%10 != 0:
                                                lenMax = round(lenMax/10)*10-10 
                                              
                                            for metricSec in secRoundNewListMetrics:   
                                                metricSec = metricSec[0:lenMax]
                                                newsecRoundNewListMetrics.append(metricSec) 
                                             
                                            secRoundNewListMetrics = newsecRoundNewListMetrics                        
                                            secRoundNewListMetricsArr = np.array([secRoundNewListMetrics])[0].T
                                            
                                            print(secRoundNewListMetricsArr.shape)
                                            sh = secRoundNewListMetricsArr.shape
                                            shx,shy = sh
                                            if shx != 0:
                                                if int(number_metrics) > len(secRoundNewListMetricsArr[0]):
                                                    number_metrics = len(secRoundNewListMetricsArr[0])
                                            else:
                                                number_metrics = shy
                                        else: 
                                            print("Not") 
                                        
                                        metric_data = secRoundNewListMetricsArr, number_metrics
                                        metricsD.append(metric_data)
                
                print("Metrics data returned: ")
                print(metricsD)
                print("\n\n\n")
                
                return metricsD, new_namesF
                                    
            #    return None
      
    
def videoAnalysisJustFeatures(curTest, numberTests, tupleForProcessing, data_from_tests): 
        
       # if curTest == 0: 
        # if True:
        #     data_from_tests = []
        
    print(tupleForProcessing)
    print(len(tupleForProcessing))
    
    if len(tupleForProcessing) == 14:
        tupleForProcessing_n = tupleForProcessing[:10] + tupleForProcessing[12:]
        tupleForProcessing = tupleForProcessing_n
        
    decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile = tupleForProcessing
    
    print("DecisorLevel: " + str(decisorLevel))
    print("MainPathVideoData: " + str(mainPathVideoData))
    print("SequenceName: " + str(sequence_name))
    print("DestinationPath: " + str(dest_path)) 
    print("MTSVideoPath: " + str(mtsVideoPath))
    print("MP4VideoFile: " + str(mp4VideoFile))
    print("IFVP: " + str(IFVP))
    print("LocationMP4File: " + str(locationMP4_file))
    print("ROIPath: " + str(roiPath))
    print("NewROIPath: " + str(newRoiPath))
    print("FirstClusteringStorageOutput: " + str(first_clustering_storing_output))
    print("PathPythonFile: " + str(pathPythonFile))
    
    
    print("\n\n --------------------- \n")
    
    print("CurTest: " + str(curTest))
    print("Number of tests: " + str(numberTests))
    
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
    
    
    
    if curTest < numberTests:    
        import time
        startTime = time.time() 
        
        
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
        
        new_mp = ""
        
        if " " in mp4_initDir or "\t" in mp4_initDir or "\n" in mp4_initDir or "\r" in mp4_initDir:
        
            for x in mp4_initDir:
                if x != " " and x != "\t" and x != "\n" and x != "\r":
                    new_mp += x
            mp4_initDir = new_mp
            
            mp4_initDir = mp4_initDir[1:]
            
            mps = mp4_initDir.split('\\')
            addFile = mps[-2]
            mx = addFile.split(':')
            
            mx_n = ""
            
            for m in mx:
                if len(m) != 0:
                    mx_n += m            
             
            mps[-2] = mx_n
            
            mp4_initDir_new = ""
            
            for x in mps[:-1]:
            
                mp4_initDir_new += x + '\\'
                
            mp4_initDir = mp4_initDir_new
        
        i = 1
    
        while True:
            if os.path.exists(mp4_initDir):
                mp4_initDir_p = mp4_initDir[:-1]
                mp4_initDir_p_x = mp4_initDir_p[:-1]
                mp4_initDir_p = mp4_initDir_p_x + str(i)
                mp4_initDir = mp4_initDir_p + "\\"
                
                i += 1
            else:
                break
        
        
        os.mkdir(mp4_initDir) 
        
        seqnew = ""
        
        for seqn in str(sequence_name):
            if seqn != " " and seqn != "\r" and seqn != "\n" and seqn != "\t" and seqn != ":":
                seqnew += seqn
                
        sequence_name_str = seqnew
        sequence_name = int(sequence_name_str)  
        
         
        
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
                    name_in = "test_" + "000" + ".avi"
                    name_out = "test_" + "00" + str(sequence_name) + ".mp4"
                    cmd = 'ffmpeg -i  ""C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/Acq_23_1_24_pseudomonas/Test_' + str(sequence_name) + '/' + name_in + "" + ' "" C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/' + name_out + ""
                    print("ffmpeg command: " + cmd)
                    
                    cmd_arr = shlex.split(cmd)
                    cmd_arr = np.array([cmd_arr])
                    cmd_arr = np.delete(cmd_arr, 3)
                    cmd = np.array([cmd_arr])[0].tolist() 
                    
                    subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=False)
                    dir_out = path_out            
                    time.sleep(5)  
                    return dir_out 
        
        if os.path.isdir(pre_dir+name_folder): 
            print("Directory already exists !!!")
        else: 
        
            newPath = os.path.join(pre_dir, name_folder)   
            
            new_path_n = ""
            
            for n in newPath:
                if n != " " and n != "\t" and n != "\r" and n != "\n" and n != "\\":
                    new_path_n += n
            
            new_path_n = new_path_n[1:]
            new_path_s = new_path_n.split('/')
            
            sp_n = ""
            
            for ind_sp, sp in enumerate(new_path_s):
                if ind_sp < len(new_path_s)-1:
                    sp_n += sp + '/'
            
            rem = new_path_s[-1]
            
            remx = rem[1:-3] + rem[-1]
            sp_n += remx
            newPath = sp_n 
            
            j = -1
            cod = 0
            
            while True:
                if '/' in newPath:
                    cod = 1
                elif "\\" in newPath:
                    cod = 2
                
                if cod == 1:
                    if os.path.exists(newPath + '/'):
                        j += 1
                        
                        if newPath[-2] == '_':
                            newPath = newPath[:-2] 
                        if newPath[-3] == '_':
                            newPath = newPath[:-3] 
                            
                        newPath = newPath + "_" + str(j)
                        print("New Path: "+ newPath)
                    else:
                        break  
                elif cod == 2:
                    if os.path.exists(newPath + "\\"):
                        j += 1
                        
                        if newPath[-2] == '_':
                            newPath = newPath[:-2] 
                        if newPath[-3] == '_':
                            newPath = newPath[:-3] 
                            
                        newPath = newPath + "_" + str(j)
                        print("New Path: "+ newPath)
                    else:
                        
                        break            
            
            if newPath[0] != 'C':
                newPath = "C" + newPath
            
            if not os.path.exists(newPath + '/'):              
                os.mkdir(newPath)  
            
            print("Directory created")
         
             
            if os.path.isfile('path_out' + '/' + str(sequence_name) + '.mp4'):
                print("MP4 File already exists inside directory") 
            else:
                print(path_in)
                
                if not os.path.isfile(path_in):
                    # Extract the number before .avi
                    file_name = os.path.basename(path_in)
                    number_before_avi = file_name.split(".")[0].split("_")[-1]
                    
                    thisN = ''
                    for n in number_before_avi:
                        if n != '\t' and n!= ':' and n.isdigit():
                            thisN += n
                    number_before_avi = str(int(thisN))
                    # if len(thisN) == 1:
                    #     number_before_avi = '00' + thisN
                    # elif len(thisN) == 2:
                    #     number_before_avi = '0' + thisN
                    
                    print("number_before_avi:")
                    print(number_before_avi) 
                 
                    # Extract the directory until GUI
                    dir_until_gui = os.path.dirname(path_in)
                    dir_until_gui = os.path.join(dir_until_gui, "Acq_23_1_24_pseudomonas")
                   # dir_until_gui += "Acq_23_1_24_pseudomonas\\"
              #      dir_until_gui = os.path.dirname(dir_until_gui)
                    
                    # Create the new file path
                    new_file_path = os.path.join(dir_until_gui, f"Test_{number_before_avi}")
                    path_in = new_file_path + "\\test_000.avi"
                print(path_in)
                print("\n---\n") 
                print(path_out)
                src_dir = loadMP4_file(path_in, path_out_mp4) 
                print("path_out_mp4: \t" + path_out_mp4)
                print("path_out: \t" + path_out)
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
            
            IFVP_n = ""
            
            for x in IFVP:
                if x != "\n" and x != "\r" and x != "\t" and x != " ":
                    IFVP_n += x
                    
            IFVP = IFVP_n  
            
            newPath = os.path.join(IFVP + str(sequence_name) + "_2")
           
            while(True): 
                
                success, image = vidcap.read()
                if success:
                    
                    if os.path.exists(newPath + '/'):
                        print("Directory already exists !!!")
                    
                    else:                      
                        newPath = os.path.join(IFVP + str(sequence_name) + "_2")
                        pathBig = IFVP + str(sequence_name) + "_2"
                        pathb = ""
                        if "\n" in pathBig or "\r" in pathBig or "\t" in pathBig or " " in pathBig:
                            print("PathBig")
                            for x in pathBig:
                                if x != "\n" and x != "\r" and x != "\t" and x != " ":
                                  pathb += x
                            if pathb[0] == ':':
                                pathb = pathb[1:]                                 
                            
                            newPath = pathb 
                         
                        if newPath[0] == ':':
                            newPath = newPath[1:]
                            
                        l = -1
                        
                        while True:
                            if os.path.exists(newPath + '/') or os.path.exists(newPath + "\\"):
                                l += 1
                                if l > 0:
                                    nps = newPath.split('/')
                                    x = nps[-1]
                                    xs = x.split('_')
                                    xs = xs[:-1]
                                    xns = "DataSequence__"
                                    
                                    for ind_ps, xps in enumerate(xs):
                                        if ind_ps >= 2:
                                            xns += xps + '_'
                                    addrem = xns[:-1]
                                    remd = ""                                    
                                    for n in nps:
                                        remd += n + '/'
                                    
                                    newPath = remd + addrem
                                    
                                newPath = newPath + "_" + str(l) 
                                print("Here for path " + str(l))
                            else:
                                break  
                         
                        os.mkdir(newPath)      
                        print("Another directory created")
                     
                    cv2.imwrite(newPath + "/video_image%d.jpg" % count, image)     
                    count += 1
                else:  
                    break           
        
            print("Images loaded") 
            
        totCountImages = count
            
        #%%
        
        roi_path = os.path.join(mainPathVideoData + roiPath + str(sequence_name) + '/')
        
        rpx = 0
        
        i = 0
        
        while True:
        
            if os.path.exists(roi_path):            
            
                if '/' in roi_path:
                    rpx = 1
                elif "\\" in roi_path:
                    rpx = 2
                    
                roi_path_p = roi_path[:-1]
                roi_path_p_x = roi_path_p[:-1]
                roi_path_p = roi_path_p_x + str(i)
                
                if rpx == 1:
                    roi_path = roi_path_p + '/'
                elif rpx == 2:
                    roi_path = roi_path_p + "\\"
               
                i += 1
            else:
                break  
        
        roi_path_clean = ""
        roi_path_clean_2 = ""
        for r in roi_path:
            if r != " " and r != "\n" and r != "\t" and r != "\r":
                roi_path_clean += r
        
        roi_path_clean = roi_path_clean[1:]
    
        for ind_r, r in enumerate(roi_path_clean):
            if not(ind_r != 1 and r == ':'):
                roi_path_clean_2 += r
            
        roi_path = roi_path_clean_2
        
        e = -1
        
        while True:
        
            if os.path.exists(roi_path + '/'):
                e += 1
                roi_pathS = roi_path.split('/')
                remRoi = roi_pathS[:-1]
                remRoiS = ""
                
                for rm in remRoi:
                    remRoiS += rm + '/'
                
                nameS = roi_pathS[-1]
                nameS_sp = nameS.split('_')
                newNameS = ""
                
                for indna, na in enumerate(nameS_sp):
                    if indna < 3:
                        newNameS += na + '_'
                newNameS += str(e)
                roi_path = remRoiS + newNameS
            else:
                break
            
        
        os.mkdir(roi_path)
     
        dec_thresh = decisorLevel     
        
        for video_image in range(0,count):        
            
                print("Analysing for image " + str(video_image) + " th") 
            
                image = cv2.imread(newPath + "/video_image%d.jpg" % video_image)        
                
                print(image)
                
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
                
                decThreshrem = "" 
                
                print("Type of dec_thresh: " + str(dec_thresh))
                
          #      if type(dec_thresh) != 'int':
                if not isinstance(dec_thresh, int):
                    for t in dec_thresh:
                        if t != '\r' and t != '\t' and t != '\n' and t != ':' and t != ' ':
                            decThreshrem += t
                    dec_thresh = decThreshrem
                     
                    dec_thresh_int = int(dec_thresh)
                else:
                    dec_thresh_int = dec_thresh
                # else:  
      #          dec_thresh_int = dec_thresh
                
                for j in range(0,len(imageOverlay[0])):
                    for i in range(0,len(imageOverlay)):
                        if imageOverlay[i,j] >= dec_thresh_int:         ### ------------------------
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
        
        rpx = 0
        
        i = 0
        
        fixed_new = ""
        fixed_new_2 = ""
        
        for f in fixed_roi_path:
            if f != "\t" and f != "\n" and f != "\r" and f != " ":
                fixed_new += f
        fixed_new = fixed_new[1:]
        
        for ind_r, r in enumerate(fixed_new):
            if not(ind_r != 1 and r == ':'):
                fixed_new_2 += r
        
        fixed_roi_path = fixed_new_2
        
        new_namesF = []
        
        while True:
        
            if os.path.exists(fixed_roi_path):            
            
                if '/' in fixed_roi_path:
                    rpx = 1
                elif "\\" in fixed_roi_path:
                    rpx = 2
                    
                fixed_roi_path_p = fixed_roi_path[:-1]
                fixed_roi_path_p_x = fixed_roi_path_p[:-1]
                fixed_roi_path_p = fixed_roi_path_p_x + str(i)
                
                if rpx == 1:
                    fixed_roi_path = fixed_roi_path_p + '/'
                elif rpx == 2:
                    fixed_roi_path = fixed_roi_path_p + "\\"
               
                i += 1
            else: 
                break       
        
            
        os.mkdir(fixed_roi_path)
        
        ## fixed_roi_path = 'C:/Research/Approach317_new' 
        
        print("totCountImages: " + str(totCountImages))
        
        if totCountImages > 0:
            write_to_excel('images_singleVideo.xlsx',curTest,totCountImages)
        else:
            numberTests -= 1
            write_number_tests_to_file(numberTests)
            
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
                
                
                
                
        from getCurrentDateAndTime import getDateTimeStrMarker          
        dateTimeMarker = getDateTimeStrMarker() 
        
        print("DecisorLevel: " + str(decisorLevel))
        print("MainPathVideoData: " + str(mainPathVideoData))
        print("SequenceName: " + str(sequence_name))
        print("DestinationPath: " + str(dest_path))
        print("MTSVideoPath: " + str(mtsVideoPath))
        print("MP4VideoFile: " + str(mp4VideoFile))
        print("IFVP: " + str(IFVP))
        print("LocationMP4File: " + str(locationMP4_file))
        print("ROIPath: " + str(roiPath))
        print("NewROIPath: " + str(newRoiPath))
        print("FirstClusteringStorageOutput: " + str(first_clustering_storing_output))
        print("PathPythonFile: " + str(pathPythonFile))
        
        data_from_tests.append((decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile, fixed_roi_path, dateTimeMarker))
        
        print("Here on test ...")
        
        print("Current test: " + str(curTest))
        print("Total number of tests: " + str(numberTests))
       
    if curTest < numberTests-1:
            print("Returning this: ")   ## 0
            for d in data_from_tests:
                for x in d:
                    print(x) 
                 
            return data_from_tests  
    elif curTest == numberTests -1 or curTest == numberTests:   ## for
    
            if curTest == numberTests:
                curTest -= 1   
     
            #%%
             
            ## - repeat the code above for all tests and compare iterativally
            ## if it's the last one, procceed. If not, save the above variables and go for the next test 
            
            print("Length for data from tests: " + str(len(data_from_tests)))    
            
            clustering_output = []
            
            metricsD = []
            
            M = 0
             
            if len(data_from_tests) > 0:
                
                if curTest == numberTests - 1 or curTest == numberTests:   ## for
                
                    if curTest == numberTests:
                        curTest -= 1
                
                    couples_combs = []
                    ## numberTests = 10
                    
                    not_include = False  
                
                    for x in range(0, numberTests):
                        for y in range(0, numberTests):
                            M += 1
                            if x != y:
                                
                                couple_number_tests = (x,y)           
                                
                                if len(couples_combs) > 0:
                                    for x in range(0, numberTests):
                                        for y in range(0, numberTests):
                                            couple_x = (x,y)
                                            if couple_number_tests[0] == couple_x[1] and couple_number_tests[1] == couple_x[0]:
                                                not_include = True
                                
                                if not not_include: 
                                    totCountImages = find_min_images('images_singleVideo.xlsx')
                                    couples_combs.append(couple_number_tests)
                           #         proc_algorithm(x, y)
                                    data_test_one = data_from_tests[x]
                                    data_test_two = data_from_tests[y]
                                    
                                    dest_path_one = data_test_one[3]
                                    dest_path_two = data_test_two[3]
                                    
                                    sequence_name_one = data_test_one[2]
                                    sequence_name_two = data_test_two[2]
                                    
                                    dateTimeMarker_one = data_test_one[-1]
                                    dateTimeMarker_two = data_test_two[-1]
                                    
                                    fixed_roi_path_one = data_test_one[-2]
                                    fixed_roi_path_two = data_test_two[-2]
                                    
                                    
                                    #######################################
                                    #######################################
                                    #######################################
                                    #######################################
                                    #######################################
                                    
                                    pathRoiStart = dest_path_one + 'modRoisFirstVideo_' + dateTimeMarker_one
                                    pathRoiEnd = dest_path_two + 'modRoisSecVideo_' + dateTimeMarker_two
                                
                                    roi_bef = pathRoiStart + str(sequence_name_one)
                                    roi_after = pathRoiEnd + str(sequence_name_two)     
                                     
                                    #%%
                                    
                                    # start_mark_before_inter = 0
                                    # end_mark_before_inter = 50
                                     
                                    # numberImagesBef = end_mark_before_inter - start_mark_before_inter + 1 
                                     
                                    # start_mark_after_inter = count-50
                                    # end_mark_after_inter = count
                                     
                                    # numberImagesAfter = end_mark_after_inter - start_mark_after_inter + 1
                                    
                                    excel_dirs = "dirs_rois_table_tries.xlsx"
                                                                      
                                    countNews = 0
                                    
                                    while True:
                                        if os.path.isfile(excel_dirs):
                                            newExcelName = ""
                                            for indDot, i in enumerate(excel_dirs):
                                                if i == '.':
                                                    newExcelName = excel_dirs[:indDot] + '_' + str(countNews) + excel_dirs[(indDot):]
                                                    excel_dirs = newExcelName
                                                    countNews += 1
                                                    break
                                        else:
                                            break 
                                    
                                    
                                    newPathA = os.path.join(roi_bef)    
                                    
                                    new_newPathA = ""
                                    
                                    for p in newPathA:
                                        if p != '\n' and p != "\n" and p!= "\t" and p != " ":
                                            new_newPathA += p
                                            
                                    new_newPathA = new_newPathA[1:]
                                    
                                    newPathA = new_newPathA
                                    
                                    itA = 0
                                    
                                    if '/' in newPathA:
                                    
                                        while os.path.exists(newPathA + '/'):
                                            
                                            if newPathA[-2] == '_':
                                                newPathA = newPathA[:-2] 
                                            if newPathA[-3] == '_':
                                                newPathA = newPathA[:-3] 
                                                
                                            newPathA += '_' + str(itA)
                                            itA += 1 
                                        else:    
                                            os.mkdir(newPathA)
                                    elif "\\" in newPathA:
                                        
                                        if newPathA[-2] == '_':
                                            newPathA = newPathA[:-2] 
                                        if newPathA[-3] == '_':
                                            newPathA = newPathA[:-3] 
                                    
                                        while os.path.exists(newPathA + "\\"):
                                            newPathA += '_' + str(itA)
                                            itA += 1 
                                        else:    
                                            os.mkdir(newPathA)
                                     
                                    newPathB = os.path.join(roi_after)   
                                    
                                    new_newPathB = ""
                                    
                                    for p in newPathB:
                                        if p != '\n' and p != "\n" and p!= "\t" and p != " ":
                                            new_newPathB += p
                                            
                                    new_newPathB = new_newPathB[1:]
                                    
                                    newPathB = new_newPathB
                                    
                                    itB = 0
                                    
                                    if '/' in newPathB:
                                    
                                        while os.path.exists(newPathB + '/'):
                                            
                                            if newPathB[-2] == '_':
                                                newPathB = newPathB[:-2] 
                                            if newPathB[-3] == '_':
                                                newPathB = newPathB[:-3] 
                                                
                                            newPathB += '_' + str(itB)
                                            itB += 1 
                                        else:    
                                            os.mkdir(newPathB)
                                    elif "\\" in newPathB:
                                    
                                        while os.path.exists(newPathB + "\\"):
                                            
                                            if newPathB[-2] == '_':
                                                newPathB = newPathB[:-2] 
                                            if newPathB[-3] == '_':
                                                newPathB = newPathB[:-3] 
                                                
                                            newPathB += '_' + str(itB)
                                            itB += 1 
                                        else:    
                                            os.mkdir(newPathB)
                                    
                           ##         os.mkdir(newPathB) 
                           
                                    write_to_excel_dirs_try(newPathA, newPathB, excel_dirs)
                                      
                                    ind_bef = 0
                                    ind_after = 0
                                    
                                    roiPathBefA = ""
                                    
                                    for p in roi_bef:
                                        if p != '\n' and p != "\n" and p!= "\t" and p != " ":
                                            roiPathBefA += p
                                    
                                    roi_bef = roiPathBefA
                                    roi_bef = roi_bef[1:]
                                    
                                    roiPathAfterB = ""
                                    
                                    for p in roi_after:
                                        if p != '\n' and p != "\n" and p!= "\t" and p != " ":
                                            roiPathAfterB += p
                                    
                                    roi_after = roiPathAfterB
                                    
                                    if roi_after[1] == 'C' and roi_after[0] != 'C':
                                        roi_after = roi_after[1:]
                                    
                                    print("Complete ROI Bef: " + roi_bef)
                                    print("Complete ROI After: " + roi_after) 
                                    
                                
                                    
                                    print("count: " + str(count))
                                    
                                    # import sys
                                    # sys.exit()
                                     
                                    for i in range(0, count):
                                        if fixed_roi_path_one[-1] == '/':
                                            fixed_roi_path_one = fixed_roi_path_one[:-1]
                                            
                                        imx = cv2.imread(fixed_roi_path_one + "/roi_image" + str(i) + ".jpg")
                                         
                                        print("Imx: ")
                                        print(imx)
                                        
                                        imxa = cv2.cvtColor(imx, cv2.COLOR_BGR2GRAY) 
                                        print(roi_bef + "/roi_image")
                                        print("Image before written")
                                        
                                        
                                        if '/' in roi_bef:
                                        
                                            if not os.path.exists(roi_bef + '/'):
                                                print("Creating new path for left ROI image")
                                                os.mkdir(roi_bef) 
                                            else:
                                                print("Path from left already exists. Overwritting ...")
                                                
                                        elif "\\" in roi_bef:
                                            if not os.path.exists(roi_bef + "\\"):
                                                print("Creating new path for left ROI image")
                                                os.mkdir(roi_bef) 
                                            else:
                                                print("Path from left already exists. Overwritting ...")
                                                                                   
                                            
                                        if not os.path.exists(fixed_roi_path_two):
                                            print("Creating new path for right ROI image")
                                            os.mkdir(fixed_roi_path_two) 
                                        else:
                                            print("Path from right already exists. Overwritting ...")
                                        
                                        
                                          
                                        cv2.imwrite(roi_bef + "/roi_image%d.jpg" % ind_bef, imxa)
                                        
                                        ind_bef += 1
                                        roi_after = roi_after[1:]
                                        
                                        
                                        if '/' in roi_after:
                                            if roi_after[0] != 'C' and roi_after[1] != ':' and roi_after[2] != '/':
                                                indU = 0
                                                ok = False
                                                for ind_r, r in enumerate(roi_after):
                                                    if r == 'U':
                                                        indU = ind_r
                                                        ok = True
                                                        break
                                                if ok:
                                                    roi_after = roi_after[indU:]
                                                
                                                print("This yes: ")
                                                print(roi_after)
                                                roi_after = 'C:/' + roi_after 
                                            if not os.path.exists(roi_after + '/'):
                                                print("Creating new path for right ROI image")
                                                os.mkdir(roi_after)  
                                            else:
                                                print("Path from left already exists. Overwritting ...")
                                                
                                        elif "\\" in roi_after: 
                                            if roi_after[0] != "C" and roi_after[1] != ":" and roi_after[2] != "\\":
                                                indU = 0
                                                ok = False
                                                for ind_r, r in enumerate(roi_after):
                                                    if r == 'U':
                                                        indU = ind_r
                                                        ok = True
                                                if ok:
                                                    roi_after = roi_after[indU:]
                                                    
                                                    
                                                roi_after = "C:\\" + roi_after
                                            if not os.path.exists(roi_after + "\\"):
                                                print("Creating new path for right ROI image")
                                                os.mkdir(roi_after) 
                                            else:
                                                print("Path from right already exists. Overwritting ...")
                                        
                                        if fixed_roi_path_two[-1] == '/':
                                            fixed_roi_path_two = fixed_roi_path_two[:-1]
                                        
                                        imx = cv2.imread(fixed_roi_path_two + "/roi_image" + str(i) + ".jpg")
                                        print(fixed_roi_path_two + "/roi_image")
                                        print("Image after written")
                                        
                                        imxa = cv2.cvtColor(imx, cv2.COLOR_BGR2GRAY) 
                                        
                                        print("Imxa: ")
                                        print(imxa)
                                        
                                        # import sys
                                        # sys.exit()
                                        
                                        if roi_after[-1] == '/':
                                            roi_after = roi_after[:-1]
                                 
                                        cv2.imwrite(roi_after + "/roi_image%d.jpg" % ind_after, imxa)
                                        
                                        ## for test ## 
                                        
                                        ime = cv2.imread(roi_after + "/roi_image" + str(ind_after) + ".jpg")
                                        print("ime: ")
                                        print(ime)
                                        
                                        # import sys
                                        # sys.exit()
                                        
                                        ##############
                                        
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
                                    
                                    for ind in range(0,int(totCountImages/2)):
                                        im1 = cv2.imread(roi_bef + "/roi_image%d.jpg" % ind)
                                        im2 = cv2.imread(roi_after + "/roi_image%d.jpg" % ind)   ##
                                        
                                        print("----- \n Image 1: \n")
                                        print(im1)
                                        
                                        print("----- \n Image 2: \n")
                                        print(im2) 
                                        
                                        ori_img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
                                        ori_img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)  
                                       
                                        couple_images = [ori_img1, ori_img2]               
                                               
                                        bigCoupleImages.append(couple_images)
                                        couple_images = []
                                
                                    print("Length for bigCoupleImages: " + str(len(bigCoupleImages))) 
                                    
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
                                
                                    print("Length for setMetricsValues: " + str(len(setMetricsValues)))
                                        
                                    newListMetrics = []
                                    
                                    for ind,  metricValue in enumerate(setMetricsValues):
                                        number_No_zeros = 0
                                        number_No_zeros = np.count_nonzero(np.array([metricValue]))
                                        
                                        print("Number no zeros: " + str(number_No_zeros))
                                        if int(number_No_zeros) > 6:     ## int(number_No_zeros) > 12
                                            newListMetrics.append(metricValue) 
                                     ##       labelsFeatQualityValues.pop(ind)
                                
                                    print("Length for newListMetrics: " + str(len(newListMetrics)))
                                    
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
                                    
                                    print("Length here for labelsFeatQualityValues: " + str(len(labelsFeatQualityValues)))
                                                                           
                                    listMetricsForEvaluation = [metric_1, metric_2, metric_3, metric_4, metric_5, metric_6, metric_7, metric_8, metric_9, metric_10, metric_11, metric_12, metric_13, metric_14, metric_15, metric_16, metric_17, metric_18, metric_19, metric_20, metric_21]
                                    
                                    time.sleep(5)
                                    
                                    newf = features_auto(labelsFeatQualityValues)
                                     
                                    print("New created functions inside extra.py file: ")
                                    print(newf)
                                    
                                    for ind_l, l in enumerate(listMetricsForEvaluation):
                                        if len(l) == 0 and ind_l < len(labelsFeatQualityValues):
                                            labelsFeatQualityValues.pop(ind_l)
                                      
                                    print("Length for listMetricsForEvaluation: " + str(len(labelsFeatQualityValues)))
                                    print("Length for each one: ")
                                    if len(listMetricsForEvaluation) > 0:
                                        for l in listMetricsForEvaluation:
                                            print(" - " + str(len(l)))
                                            
                                    print("Length A for labelsFeatQualityValues: " + str(len(labelsFeatQualityValues)))
                                        
                                    indsConst = []
                                                                        
                                    for ind_metricConstant, newMetricEval in enumerate(listMetricsForEvaluation):        
                                                if len(np.unique(np.array([newMetricEval]))) == 1:
                                             #       print("Constant feature detected")              
                                                    not_constant_Flag += 1           
                                                    listMetricsForEvaluation.pop(ind_metricConstant)               
                                                    indsConst.append(ind_metricConstant)
                                                    labelsFeatQualityValues.pop(ind_metricConstant)
                                         
                                    print("Length B for labelsFeatQualityValues: " + str(len(labelsFeatQualityValues)))
                                    
                                    secRoundNewListMetrics = []
                                    metricsIndicesDeleted = [] 
                                     
                                    number_metrics = 20
                                                
                                    if True:
                                        for ind_new_sec, newMetricEval in enumerate(listMetricsForEvaluation):
                                            number_No_zeros = 0
                                            number_No_zeros = np.count_nonzero(np.array([newMetricEval]))
                                            
                                            print("Preliminar metric eval.: " + str(len(newMetricEval)))
                                             
                                            if number_No_zeros > len(newMetricEval)/2:   ## number_No_zeros > len(newMetricEval)/2
                                                secRoundNewListMetrics.append(newMetricEval)
                                            else: 
                                                metricsIndicesDeleted.append(ind_new_sec)
                                          #      labelsFeatQualityValues.pop(ind_new_sec)
                                    
                                    
                                    print("Indices deleted: ")
                                    for i in metricsIndicesDeleted:
                                        print(i)
                                        
                                    metricsIndD_inv = metricsIndicesDeleted[::-1]
                                    
                                    for m in metricsIndD_inv:
                                        labelsFeatQualityValues.pop(m)
                                    
                                    print("\n\n\n")
                                    
                                    # for ind_d, d in enumerate(labelsFeatQualityValues):
                                    #     if ind_d not in metricsIndicesDeleted: 
                                    #         new_namesF.append(d)
                                    
                                    new_namesF = labelsFeatQualityValues
                                    
                                    print("Length of new_namesF: " + str(len(new_namesF)))
                                    
                                    test_size = 0.2
                                    newsecRoundNewListMetrics = []
                                    
                                    
                           #         print("Metrics here on source: ")
                           #         print(secRoundNewListMetrics)
                                     
                                    lenMax = 0
                                    
                                    if len(secRoundNewListMetrics) > 0:
                                    
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
                                    else:
                                        print("Not") 
                                    
                                    metric_data = secRoundNewListMetricsArr, number_metrics
                                    metricsD.append(metric_data)
            # print("Metrics data returned: ")
            # print(metricsD)
            # print("\n\n\n") 
                                        
            # return metricsD, new_namesF   
                 
            print("Metrics data returned: ")
            print(metricsD)
            print("\n\n\n") 
                                                
            return metricsD, new_namesF     
