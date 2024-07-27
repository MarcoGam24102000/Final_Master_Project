# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 00:05:15 2022

@author: marco
"""

from subVideoAnalysis import write_to_excel, find_min_images


def write_number_tests_to_file(numberTests):
    with open('temp_numberTests.txt', 'a') as file:
        file.write(str(numberTests))
        
def videoAnalysisMultipleVideosWithinOne(curTest, numberTests, tupleForProcessing, data_from_tests, direcPythonFile, filename_output_video, fps_out, limToCompare, buffer_imgs):
        
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
        
        if curTest == 1:
        
            print("Length of buffer_imgs: " + str(len(buffer_imgs)))
            print("Length of buffer of last test:" + str(len(buffer_imgs[1])))
            
            # import sys
            # sys.exit()
        
        buffer_imgsx = buffer_imgs[curTest]
            
        filename_output_video += '.mp4'   
        
        limToCompare = 50
        
       
        
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
                    
                     while True:
                    
                         fixed_roi_path = fixed_roi_path[:lastF] + str(ind) + '/'
                        
                         if not os.path.exists(fixed_roi_path):
                    
                             os.mkdir(fixed_roi_path)
                             break
                         
                     print("\n\n\n ---------- \n\n Fixed ROI Path before: \n\n\n ---------- \n\n")
                     print(fixed_roi_path)
                    
                     # import sys
                     # sys.exit()
                    
                     ## fixed_roi_path = 'C:/Research/Approach317_new' 
                    
                     print("totCountImages: " + str(totCountImages))
                    
                     print("\n\n\n ---------- \n\n Fixed ROI Path before: \n\n\n ---------- \n\n")
                     print(fixed_roi_path)
                     
                     if fixed_roi_path[-1] == '/':
                         fixed_roi_path = fixed_roi_path[:-1]
                    
                     if totCountImages > 0:
                         write_to_excel('images_singleVideo.xlsx',curTest,totCountImages)
                     else:
                   #      numberTests -= 1
                         write_number_tests_to_file(numberTests)
                         
                     print("totCountImages:" + str(totCountImages))
                     
                     print(fixed_roi_path + "/roi_image%d.jpg")
                     
                     for video_image in range(0,totCountImages):  
                         
                             print("Analysing for roi " + str(video_image) + " th")
                        
                             image = cv2.imread(roi_path + "/roi_image%d.jpg"  % video_image)        
                             imx = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
                            
                             x_dim = len(imx)
                             y_dim = len(imx[0])        
                             x_center = int(x_dim/2) 
                             y_center = int(y_dim/2)        
                             imd = imx[x_center-50:x_center+50, y_center-50:y_center+50]
                            
                             s = cv2.imwrite(fixed_roi_path + "/roi_image%d.jpg" % video_image, imd)         
                             print(s)
                             count += 1 
                    
                     
                     # import sys
                     # sys.exit()
                     
                     time.sleep(10)
                            
                            
                            
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
                     
                     # if curTest == numberTests-1:
                     #   import sys
                     #   sys.exit()
               
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
                                         
                               ##         if secRoundNewListMetricsArr:
                                        if True:
                                            
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
                                            
                                            print("Lists: ")
                                            print(list1)
                                            print(list2)
                                            print(list3)
                                            print(list4)
                                            print(list5)
                                            print(list6)                                     
                                            
                                            indForFolderClustering = []    
                                            
                                            list1FirstArr = np.array([np.array([list1])])    
                                            listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
                                            
                                            print("\n\n ------- \nListArrToCompare: ")
                                            print(listArrToCompare)
                                            print("-------\n\n") 
                                            
                                            print("\n\n ------- \nList1FirstArray: ")
                                            print(list1FirstArr)
                                            print("-------\n\n")
                                            
                                            print("\n\n ------- \nsecRoundNewListMetricsArr: ")
                                            print(secRoundNewListMetricsArr)
                                            print("-------\n\n")
                                            
                                            print(len(list1FirstArr[0][0]))
                                            print("Iterating ...")
                                            
                                            singIndData = []
                                            
                                            for ind in range(0,len(list1FirstArr[0][0])):       ## 
                                                
                                                val = []
                                                
                                                for xi in list1FirstArr[0][0]:
                                                    if ind < len(xi):
                                                        val.append(xi[ind])
                                                    
                                                print("First value: ")
                                                print(secRoundNewListMetricsArr[ind])
                                                
                                                print("Length:")
                                                print(len(secRoundNewListMetricsArr[ind]))
                                                
                                                new_strSecRound = ""
                                                secRoundList = []
                                                
                                                if len(secRoundNewListMetricsArr[ind]) == 1:
                                                   print("String")
                                                   new_strSecRound = str(secRoundNewListMetricsArr[ind][0]) 
                                                   print(new_strSecRound)
                                                else:
                                                    for x in secRoundNewListMetricsArr[ind]:
                                                        secRoundList.append(x)
                                                    print("secRoundList: ")
                                                    print(secRoundList)
                                                    
                                              
                                                print("Value to compare to: ")
                                                print(val)
                                                print("list1 first array: ")
                                                print(list1FirstArr[0][0]) 
                                                
                                                if len(val) == len(secRoundList)-1:
                                                    secRoundList = secRoundList[:-1]
                                                
                                           ##     tupleIndiceImage = np.where(secRoundList == val)    ## [0,ind]
                                                
                                                ind_data = []
                                                for x in range(len(val)):
                                                    if val[x] == secRoundList[x]:
                                                       ind_data.append(x) 
                                                       
                                         ##       rub, ind_data = tupleIndiceImage
                                           #     ind_data = np.array([ind_data])
                                                
                                                print("Ind data for 1, for ind " + str(ind) + " : ")
                                                print(ind_data)
                                               
                                                # if len(ind_data[0]) == 1:
                                                #     print("Appending indice " + str(ind_data) + " to singIndData")
                                                #     singIndData.append(ind_data) 
                                                # else:
                                                #     if len(ind_data[0]) == 2:
                                                #         print("Not singular")
                                                #         singIndData.append(ind_data[0,0])
                                                #         singIndData.append(ind_data[0,1]) 
                                                
                                                print("Length ind data: " + str(len(ind_data)))
                                                
                                                if len(ind_data) > 0:
                                                    
                                                    for i in range(0,len(ind_data)):
                                                    
                                                        singIndData.append(ind_data[i])
                                            
                                                    # ind_datax = ind_data[0]  
                                                    # newDat = []
                                                    
                                                    # print("ind_data: ")
                                                    # print(ind_data)
                                                    
                                                    # for i in ind_datax:
                                                    #     newDat.append(i)
                                                    
                                                    # singIndData.append(newDat[0])
                                                    # singIndData.append(newDat[1])
                                            
                                            newSingData = []
                                            
                                            print("Sing Ind Data:")
                                            print(singIndData)
                                            
                                            if len(singIndData) > 0:
                                            
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
                                            
                                            list2FirstArr = np.array([np.array([list2])])    
                                            listArrToCompare = np.array([secRoundNewListMetricsArr[:,1]])
                                            
                                            print("\n\n ------- \nListArrToCompare: ")
                                            print(listArrToCompare)
                                            print("-------\n\n") 
                                            
                                            print("\n\n ------- \nList1FirstArray: ")
                                            print(list2FirstArr)
                                            print("-------\n\n")
                                            
                                            print("\n\n ------- \nsecRoundNewListMetricsArr: ")
                                            print(secRoundNewListMetricsArr)
                                            print("-------\n\n")
                                            
                                            print(len(list2FirstArr[0][0]))
                                            print("Iterating ...")
                                            
                                            singIndData = []
                                            
                                            for ind in range(0,len(list2FirstArr[0][0])):       ## 
                                                
                                                val = []
                                                
                                                for xi in list2FirstArr[0][0]:
                                                    if ind < len(xi):
                                                        val.append(xi[ind])
                                                    
                                                print("First value: ")
                                                print(secRoundNewListMetricsArr[ind])
                                                
                                                print("Length:")
                                                print(len(secRoundNewListMetricsArr[ind]))
                                                
                                                new_strSecRound = ""
                                                secRoundList = []
                                                
                                                if len(secRoundNewListMetricsArr[ind]) == 1:
                                                   print("String")
                                                   new_strSecRound = str(secRoundNewListMetricsArr[ind][0]) 
                                                   print(new_strSecRound)
                                                else:
                                                    for x in secRoundNewListMetricsArr[ind]:
                                                        secRoundList.append(x)
                                                    print("secRoundList: ")
                                                    print(secRoundList)
                                                    
                                              
                                                print("Value to compare to: ")
                                                print(val)
                                                print("list2 first array: ")
                                                print(list2FirstArr[0][0]) 
                                                
                                                if len(val) == len(secRoundList)-1:
                                                    secRoundList = secRoundList[:-1]
                                                
                                           ##     tupleIndiceImage = np.where(secRoundList == val)    ## [0,ind]
                                                
                                                ind_data = []
                                                for x in range(len(val)):
                                                    if val[x] == secRoundList[x]:
                                                       ind_data.append(x) 
                                                       
                                         ##       rub, ind_data = tupleIndiceImage
                                        #        ind_data = np.array([ind_data])
                                                
                                                print("Ind data for 2, for ind " + str(ind) + " : ")
                                                print(ind_data)
                                               
                                                # if len(ind_data[0]) == 1:
                                                #     print("Appending indice " + str(ind_data) + " to singIndData")
                                                #     singIndData.append(ind_data) 
                                                # else:
                                                #     if len(ind_data[0]) == 2:
                                                #         print("Not singular")
                                                #         singIndData.append(ind_data[0,0])
                                                #         singIndData.append(ind_data[0,1])           
                                            
                                                print("Length ind data: " + str(len(ind_data)))
                                               
                                                if len(ind_data) > 0:
                                                   
                                                   for i in range(0,len(ind_data)):
                                                   
                                                       singIndData.append(ind_data[i])
                                            
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
                                            
                                            list3FirstArr = np.array([np.array([list3])])    
                                            listArrToCompare = np.array([secRoundNewListMetricsArr[:,2]])
                                            
                                            print("\n\n ------- \nListArrToCompare: ")
                                            print(listArrToCompare)
                                            print("-------\n\n") 
                                            
                                            print("\n\n ------- \nList3FirstArray: ")
                                            print(list3FirstArr)
                                            print("-------\n\n")
                                            
                                            print("\n\n ------- \nsecRoundNewListMetricsArr: ")
                                            print(secRoundNewListMetricsArr)
                                            print("-------\n\n")
                                            
                                            print(len(list3FirstArr[0][0]))
                                            print("Iterating ...")
                                            
                                            singIndData = []
                                            
                                            for ind in range(0,len(list3FirstArr[0][0])):       ## 
                                                
                                                val = []
                                                
                                                for xi in list3FirstArr[0][0]:
                                                    if ind < len(xi):
                                                        val.append(xi[ind])
                                                    
                                                print("First value: ")
                                                print(secRoundNewListMetricsArr[ind])
                                                
                                                print("Length:")
                                                print(len(secRoundNewListMetricsArr[ind]))
                                                
                                                new_strSecRound = ""
                                                secRoundList = []
                                                
                                                if len(secRoundNewListMetricsArr[ind]) == 1:
                                                   print("String")
                                                   new_strSecRound = str(secRoundNewListMetricsArr[ind][0]) 
                                                   print(new_strSecRound)
                                                else:
                                                    for x in secRoundNewListMetricsArr[ind]:
                                                        secRoundList.append(x)
                                                    print("secRoundList: ")
                                                    print(secRoundList)
                                                    
                                              
                                                print("Value to compare to: ")
                                                print(val)
                                                print("list3 first array: ")
                                                print(list3FirstArr[0][0]) 
                                                
                                                if len(val) == len(secRoundList)-1:
                                                    secRoundList = secRoundList[:-1]
                                                
                                           ##     tupleIndiceImage = np.where(secRoundList == val)    ## [0,ind]
                                                
                                                ind_data = []
                                                for x in range(len(val)):
                                                    if val[x] == secRoundList[x]:
                                                       ind_data.append(x) 
                                                       
                                         ##       rub, ind_data = tupleIndiceImage
                                          ##      ind_data = np.array([ind_data])
                                                
                                                print("Ind data for 3, for ind " + str(ind) + " : ")
                                                print(ind_data)
                                               
                                                # if len(ind_data[0]) == 1:
                                                #     print("Appending indice " + str(ind_data) + " to singIndData")
                                                #     singIndData.append(ind_data) 
                                                # else:
                                                #     if len(ind_data[0]) == 2:
                                                #         print("Not singular")
                                                #         singIndData.append(ind_data[0,0])
                                                #         singIndData.append(ind_data[0,1])           
                                            
                                                print("Length ind data: " + str(len(ind_data)))
                                               
                                                if len(ind_data) > 0:
                                                   
                                                   for i in range(0,len(ind_data)):
                                                   
                                                       singIndData.append(ind_data[i])
                                                
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
                                            
                                            list4FirstArr = np.array([np.array([list4])])    
                                            listArrToCompare = np.array([secRoundNewListMetricsArr[:,3]])
                                            
                                            print("\n\n ------- \nListArrToCompare: ")
                                            print(listArrToCompare)
                                            print("-------\n\n") 
                                            
                                            print("\n\n ------- \nList4FirstArray: ")
                                            print(list4FirstArr)
                                            print("-------\n\n")
                                            
                                            print("\n\n ------- \nsecRoundNewListMetricsArr: ")
                                            print(secRoundNewListMetricsArr)
                                            print("-------\n\n")
                                            
                                            print(len(list4FirstArr[0][0]))
                                            print("Iterating ...")
                                            
                                            singIndData = []
                                            
                                            for ind in range(0,len(list4FirstArr[0][0])):       ## 
                                                
                                                val = []
                                                
                                                for xi in list4FirstArr[0][0]:
                                                    if ind < len(xi):
                                                        val.append(xi[ind])
                                                    
                                                print("First value: ")
                                                print(secRoundNewListMetricsArr[ind])
                                                
                                                print("Length:")
                                                print(len(secRoundNewListMetricsArr[ind]))
                                                
                                                new_strSecRound = ""
                                                secRoundList = []
                                                
                                                if len(secRoundNewListMetricsArr[ind]) == 1:
                                                   print("String")
                                                   new_strSecRound = str(secRoundNewListMetricsArr[ind][0]) 
                                                   print(new_strSecRound)
                                                else:
                                                    for x in secRoundNewListMetricsArr[ind]:
                                                        secRoundList.append(x)
                                                    print("secRoundList: ")
                                                    print(secRoundList)
                                                    
                                              
                                                print("Value to compare to: ")
                                                print(val)
                                                print("list4 first array: ")
                                                print(list4FirstArr[0][0]) 
                                                
                                                if len(val) == len(secRoundList)-1:
                                                    secRoundList = secRoundList[:-1]
                                                
                                           ##     tupleIndiceImage = np.where(secRoundList == val)    ## [0,ind]
                                                
                                                ind_data = []
                                                for x in range(len(val)):
                                                    if val[x] == secRoundList[x]:
                                                       ind_data.append(x) 
                                                       
                                         ##       rub, ind_data = tupleIndiceImage
                                         ##       ind_data = np.array([ind_data])
                                                
                                                print("Ind data for 4, for ind " + str(ind) + " : ")
                                                print(ind_data)
                                               
                                                # if len(ind_data[0]) == 1:
                                                #     singIndData.append(ind_data)
                                                #     print("Appending indice " + str(ind_data) + " to singIndData")
                                                # else:
                                                #     if len(ind_data[0]) == 2:
                                                #         print("Not singular")
                                                #         singIndData.append(ind_data[0,0])
                                                #         singIndData.append(ind_data[0,1])           
                                            
                                                print("Length ind data: " + str(len(ind_data)))
                                               
                                                if len(ind_data) > 0:
                                                   
                                                   for i in range(0,len(ind_data)):
                                                   
                                                       singIndData.append(ind_data[i])
                                            
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
                                            
                                            list5FirstArr = np.array([np.array([list5])])    
                                            listArrToCompare = np.array([secRoundNewListMetricsArr[:,4]])
                                            
                                            print("\n\n ------- \nListArrToCompare: ")
                                            print(listArrToCompare)
                                            print("-------\n\n") 
                                            
                                            print("\n\n ------- \nList5FirstArray: ")
                                            print(list5FirstArr)
                                            print("-------\n\n")
                                            
                                            print("\n\n ------- \nsecRoundNewListMetricsArr: ")
                                            print(secRoundNewListMetricsArr)
                                            print("-------\n\n")
                                            
                                            print(len(list5FirstArr[0][0]))
                                            print("Iterating ...")
                                            
                                            singIndData = []
                                            
                                            for ind in range(0,len(list5FirstArr[0][0])):       ## 
                                                
                                                val = []
                                                
                                                for xi in list5FirstArr[0][0]:
                                                    if ind < len(xi):
                                                        val.append(xi[ind])
                                                    
                                                print("First value: ")
                                                print(secRoundNewListMetricsArr[ind])
                                                
                                                print("Length:")
                                                print(len(secRoundNewListMetricsArr[ind]))
                                                
                                                new_strSecRound = ""
                                                secRoundList = []
                                                
                                                if len(secRoundNewListMetricsArr[ind]) == 1:
                                                   print("String")
                                                   new_strSecRound = str(secRoundNewListMetricsArr[ind][0]) 
                                                   print(new_strSecRound)
                                                else:
                                                    for x in secRoundNewListMetricsArr[ind]:
                                                        secRoundList.append(x)
                                                    print("secRoundList: ")
                                                    print(secRoundList)
                                                    
                                              
                                                print("Value to compare to: ")
                                                print(val)
                                                print("list5 first array: ")
                                                print(list5FirstArr[0][0]) 
                                                
                                                if len(val) == len(secRoundList)-1:
                                                    secRoundList = secRoundList[:-1]
                                                
                                           ##     tupleIndiceImage = np.where(secRoundList == val)    ## [0,ind]
                                                
                                                ind_data = []
                                                for x in range(len(val)):
                                                    if val[x] == secRoundList[x]:
                                                       ind_data.append(x) 
                                                       
                                         ##       rub, ind_data = tupleIndiceImage
                                         ##       ind_data = np.array([ind_data])
                                                
                                                print("Ind data for 5, for ind " + str(ind) + " : ")
                                                print(ind_data)
                                                
                                                print("Length ind data: " + str(len(ind_data)))
                                               
                                                if len(ind_data) > 0:
                                                   
                                                   for i in range(0,len(ind_data)):
                                                   
                                                       singIndData.append(ind_data[i])
                                               
                                                # if len(ind_data[0]) == 1:
                                                #     print("Appending indice " + str(ind_data) + " to singIndData")
                                                #     singIndData.append(ind_data)  
                                                # else:
                                                #     if len(ind_data[0]) == 2:
                                                #         print("Not singular")
                                                #         singIndData.append(ind_data[0,0])
                                                #         singIndData.append(ind_data[0,1])           
                                            
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
                                                
                                            list6FirstArr = np.array([np.array([list6])])    
                                            listArrToCompare = np.array([secRoundNewListMetricsArr[:,5]])
                                            
                                            print("\n\n ------- \nListArrToCompare: ")
                                            print(listArrToCompare)
                                            print("-------\n\n") 
                                            
                                            print("\n\n ------- \nList6FirstArray: ")
                                            print(list6FirstArr)
                                            print("-------\n\n")
                                            
                                            print("\n\n ------- \nsecRoundNewListMetricsArr: ")
                                            print(secRoundNewListMetricsArr)
                                            print("-------\n\n")
                                            
                                            print(len(list6FirstArr[0][0]))
                                            print("Iterating ...")
                                            
                                            singIndData = []
                                            
                                            for ind in range(0,len(list6FirstArr[0][0])):       ## 
                                                
                                                val = []
                                                
                                                for xi in list5FirstArr[0][0]:
                                                    if ind < len(xi):
                                                        val.append(xi[ind])
                                                    
                                                print("First value: ")
                                                print(secRoundNewListMetricsArr[ind])
                                                
                                                print("Length:")
                                                print(len(secRoundNewListMetricsArr[ind]))
                                                
                                                new_strSecRound = ""
                                                secRoundList = []
                                                
                                                if len(secRoundNewListMetricsArr[ind]) == 1:
                                                   print("String")
                                                   new_strSecRound = str(secRoundNewListMetricsArr[ind][0]) 
                                                   print(new_strSecRound)
                                                else:
                                                    for x in secRoundNewListMetricsArr[ind]:
                                                        secRoundList.append(x)
                                                    print("secRoundList: ")
                                                    print(secRoundList)
                                                    
                                              
                                                print("Value to compare to: ")
                                                print(val)
                                                print("list6 first array: ")
                                                print(list6FirstArr[0][0]) 
                                                
                                                if len(val) == len(secRoundList)-1:
                                                    secRoundList = secRoundList[:-1]
                                                
                                           ##     tupleIndiceImage = np.where(secRoundList == val)    ## [0,ind]
                                                
                                                ind_data = []
                                                for x in range(len(val)):
                                                    if val[x] == secRoundList[x]:
                                                       ind_data.append(x) 
                                                       
                                         ##       rub, ind_data = tupleIndiceImage
                                         ##       ind_data = np.array([ind_data])
                                                
                                                print("Ind data for 6, for ind " + str(ind) + " : ")
                                                print(ind_data)
                                                
                                                print("Length ind data: " + str(len(ind_data)))
                                               
                                                if len(ind_data) > 0:
                                                   
                                                   for i in range(0,len(ind_data)):
                                                   
                                                       singIndData.append(ind_data[i])
                                               
                                                # if len(ind_data[0]) == 1:
                                                #     print("Appending indice " + str(ind_data) + " to singIndData")
                                                #     singIndData.append(ind_data) 
                                                # else:
                                                #     if len(ind_data[0]) == 2:
                                                #         print("Not singular")
                                                #         singIndData.append(ind_data[0,0])
                                                #         singIndData.append(ind_data[0,1])           
                                            
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
                                              
                                            print("\n\n-------------")
                                            print("Ind for folder clustering: ")
                                            print(indForFolderClustering)
                                            print("-------------\n\n")
                                                
                                            #%%
                                            
                                            
                                            ###############################################################################################################################################
                                            ###############################################################################################################################################
                                            ###############################################################################################################################################
                                            ###############################################################################################################################################
                                            ###############################################################################################################################################
                                        
                                            ###############################################################################################################################################
                                            ###############################################################################################################################################
                                            ###############################################################################################################################################
                                            ###############################################################################################################################################
                                            ###############################################################################################################################################
                                        
                                            name_folder = first_clustering_storing_output + str(sequence_name) + "_7_8"
                                            newPath = os.path.join(parent_dir,name_folder)    
                                            
                                            
                                            if "\n" in newPath or "\t" in newPath or "\r" in newPath or " " in newPath:                                      
                                            
                                                n_path = ""
                                                
                                                for let in newPath:
                                                    if let != "\n" and let != "\r" and l != "\t" and l != " ":
                                                        n_path += let
                                                
                                                print("NewPath Bef: " + str(n_path))
                                                n_path = n_path[1:]
                                                newPath = n_path
                                                
                                                print("NewPath After: " + str(newPath))
                                                
                                                if '/' in newPath:
                                                    x = 0
                                                    while os.path.exists(newPath + '/'):
                                                        newPath += '_' + str(x)
                                                        x += 1
                                                    nPath = newPath.split('/')
                                                    pas = nPath[-1]
                                                    rpas = nPath[:-1]
                                                    pas = pas[2:]
                                                    pasn = ""
                                                    for p in pas:
                                                        if p != "\t" and p != " ":
                                                            pasn += p
                                                    pas = pasn
                                                    rpas.append(pas)
                                                    nnPath = ""
                                                    for r in rpas:
                                                        nnPath += r + '/'
                                                    
                                                    nnPath = nnPath[:-1]
                                                    newPath = nnPath
                                                    
                                                elif "\\" in newPath:
                                                    if os.path.exists(newPath + "\\"):
                                                        x = 0
                                                        while os.path.exists(newPath + "\\"):
                                                            newPath += '_' + str(x)
                                                            x += 1
                                                        nPath = newPath.split('/')
                                                        pas = nPath[-1]
                                                        rpas = nPath[:-1]
                                                        pas = pas[2:]
                                                        pasn = ""
                                                        for p in pas:
                                                          if p != "\t" and p != " ":
                                                             pasn += p
                                                        pas = pasn
                                                        rpas.append(pas)
                                                        nnPath = ""
                                                        for r in rpas:
                                                           nnPath += r + '/'
                                                           
                                                        nnPath = nnPath[:-1] 
                                                        newPath = nnPath
                                                
                                                indxp = 0
                                                for indp, p in enumerate(newPath):
                                                    if indp < 10:
                                                        if p == "\n" or p == ' ' or p == "\t":
                                                            indxp = indp
                                                            
                                                newPath = newPath[indxp+1:]
                                                
                                                if os.path.exists(newPath + "\\"):
                                                    x = 0
                                                    while os.path.exists(newPath + "\\"):
                                                        
                                                        if newPath[-2] == '_':
                                                            newPath = newPath[:-2] 
                                                        if newPath[-3] == '_':
                                                            newPath = newPath[:-3] 
                                                            
                                                        newPath += '_' + str(x)
                                                        x += 1
                                                    else:
                                                        os.mkdir(newPath) 
                                            else:
                                                os.mkdir(newPath) 
                                            
                                            if not os.path.exists(newPath + "/"):
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
                                            
                                            
                                            print("Cluster_ID_transpose: ")
                                            print(Cluster_ID_transpose)          
                                            
                                            for ind_imageInCluster, cluster in enumerate(Cluster_ID_transpose[0]):    ## indForFolderClustering
                                              ##  cluster_list = cluster.tolist()
                                             ##   for ind_imageInCluster in cluster_list[0]:
                                            
                                                    image_counter = cv2.imread(roi_bef + "/roi_image" + str(ind_imageInCluster) + ".jpg")
                                                    image_counter_2 = cv2.imread(roi_after + "/roi_image" + str(ind_imageInCluster) + ".jpg")       
                                                    
                                                    if cluster == 0:  
                                                        cv2.imwrite(newPath_1 + "/" + "image_" + str(counter_1) + "_" + str(0) + ".jpg", image_counter) 
                                                        cv2.imwrite(newPath_1 + "/" + "image_" + str(counter_1) + "_" + str(1) + ".jpg", image_counter_2)
                                                        metricsIdTtrain1.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                                                        counter_1 += 1
                                                    if cluster == 1:
                                                        cv2.imwrite(newPath_2 + "/" + "image_" + str(counter_2) + "_" + str(0) + ".jpg", image_counter)
                                                        cv2.imwrite(newPath_2 + "/" + "image_" + str(counter_2) + "_" + str(1) + ".jpg", image_counter_2)
                                                        metricsIdTtrain2.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                                                        counter_2 += 1
                                                    if cluster == 2: 
                                                        cv2.imwrite(newPath_3 + "/" + "image_" + str(counter_3) + "_" + str(0) + ".jpg", image_counter)
                                                        cv2.imwrite(newPath_3 + "/" + "image_" + str(counter_3) + "_" + str(1) + ".jpg", image_counter_2)
                                                        metricsIdTtrain3.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                                                        counter_3 += 1 
                                                    if cluster == 3:
                                                        cv2.imwrite(newPath_4 + "/" + "image_" + str(counter_4) + "_" + str(0) + ".jpg", image_counter)
                                                        cv2.imwrite(newPath_4 + "/" + "image_" + str(counter_4) + "_" + str(1) + ".jpg", image_counter_2)
                                                        metricsIdTtrain4.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                                                        counter_4 += 1
                                                    if cluster == 4:
                                                        cv2.imwrite(newPath_5 + "/" + "image_" + str(counter_5) + "_" + str(0) + ".jpg", image_counter)
                                                        cv2.imwrite(newPath_5 + "/" + "image_" + str(counter_5) + "_" + str(1) + ".jpg", image_counter_2)
                                                        metricsIdTtrain5.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                                                        counter_5 += 1
                                                    if cluster == 5:
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
                                            
                                            # for ind_mat1, mat1 in enumerate(metrics_afterClustering):
                                            #     mat1_n = []
                                            #     for ind_mat2, mat2 in enumerate(mat1):
                                            #         mat2_n = [] 
                                            #         for ind_mat3, mat3 in enumerate(mat2):
                                            #             mat3 = mat3[0]
                                            #             mat2_n.append(mat3)
                                            #         mat1_n.append(mat2_n)  
                                                
                                            #     metricsToPCA_analysis.append(mat1_n)
                                             
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
                                            
                                            dfxi = dfxi.dropna()
                                            dfxi = dfxi.dropna(axis=1)
                                            
                                            print("Dataframe: ") 
                                            print(dfxi.to_string())   
                                            
                                            pcai = PCA(n_components=None)                             
                                            
                                            dfx_pcai = pcai.fit(dfxi)      ## Error
                                             
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
                                            
                                            print("\nClass A: ")
                                            print(classAFurther)
                                            print("\nClass B: ")
                                            print(classBFurther)
                                            
                                            if not(len(classAFurther) < 2 or len(classBFurther) < 2):
                                                        
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
                                                
                                                print("\n dfx_transi1: ")
                                                print(dfx_transi1)
                                                print("\n dfx_transi2: ")
                                                print(dfx_transi2)
                                                
                                                # import sys
                                                # sys.exit()
                                                
                                                if len(dfx_transi1) == 2 and len(dfx_transi2) == 2:
                                                
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
                                            else:
                                                import PySimpleGUI as sg
                                                
                                                sg.popup('Only possible to form one cluster. Not showing any data ... ', title = 'Clustering issue !')
                                                
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
                                                    print("The number of recommended clusters (from clustering approach) is higher than the number of clusters computed from dist-linkage method.")
                                                else:
                                                    if number_recommended_clusters < numberClustersFromDist:
                                                        print("The number of clusters computed from dist-linkage method is higher than the number of recommended clusters (from clustering approach).")
                                            
                                            
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
                                            
                                            ind_slash = 0
                                            
                                            for ind_x, x in enumerate(IFVP):
                                                if x == '/':
                                                    ind_slash = ind_x
                                                    break
                                            
                                            IFVP_p = IFVP[ind_slash:]
                                            IFVP = 'C:' + IFVP_p
                                                    
                                            
                                            # if IFVP[0] != 'C' and IFVP[0] == ':':
                                            #     IFVP = 'C' + IFVP
                                            # elif IFVP[0] != 'C' and  IFVP[0] != ':':
                                            #     IFVP = IFVP[1:]
                                            
                                            print(IFVP + str(sequence_name) + "_2" + "/video_image")
                                            
                                            for video_image in range(0,count):           #### 321   ## from 361   ## to count = 637
                                                
                                                    print("Analysing for image " + str(video_image) + " th")
                                                
                                                    image = cv2.imread(IFVP + str(sequence_name) + "_2" + "/video_image%d.jpg" % video_image)        
                                                    
                                                    if image is not None:
                                                        imx = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
                                                       
                                                   
                                                     
                                                        M1 = convolve2D(imx, k1)
                                                        M2 = convolve2D(imx, k2)      
                                                        
                                                        
                                                        print("Generating output for image " + str(video_image))
                                                        
                                                        comp_x = np.power(np.array([M1])[0], 2)      
                                                        comp_y = np.power(np.array([M2])[0], 2) 
                                                        
                                                        sum_comps = comp_x + comp_y 
                                                        
                                                        gen_output = np.power(sum_comps, 1/2).astype(int)
                                                        
                                                        cv2.imwrite(roi_path + "/gen_output_%d.jpg" % video_image, gen_output)
                                                        
                                                #######################################
                                                #######################################
                                                #######################################
                                                #######################################
                                                #######################################  
                                                 
                                                        executionTime = (time.time() - startTime)
                                                        print('Execution time in seconds: ' + str(executionTime))
                                                    else:
                                                        break
                                                
                                            clustering_output.append((clustering_inf_data, executionTime, totCountImages))
                                             
                                            
                                        else:
                                            not_include = True     
                     
                 
                    
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
            
            
               
        ##    return percIntervalForK, executionTime, totCountImages

                            print("M: " + str(M))                   
                        
                            if len(clustering_output) == 1:
                                clustering_output = clustering_output[0]
                            
                            print("Returning clustering output")
                            print(clustering_output)
                            
                            time.sleep(30)
                            return clustering_output   
                                            
            #    return None
     
    
def videoAnalysis(curTest, numberTests, tupleForProcessing, data_from_tests): 
        
       # if curTest == 0:
        # if True:
        #     data_from_tests = []
        
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
                if True:
                    if not isinstance(dec_thresh, int):
                        for t in dec_thresh:
                            if t != '\r' and t != '\t' and t != '\n' and t != ':' and t != ' ':
                                decThreshrem += t
                        dec_thresh = decThreshrem
                         
                        dec_thresh_int = int(dec_thresh) 
                    else:
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
        
            
        os.mkdir(fixed_roi_path)
        
        ## fixed_roi_path = 'C:/Research/Approach317_new' 
        
        print("totCountImages: " + str(totCountImages))
        
        for video_image in range(0,totCountImages):  
            
                print("Analysing for roi " + str(video_image) + " th")
            
                image = cv2.imread(roi_path + "/roi_image%d.jpg"  % video_image)        
                imx = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
                
                x_dim = len(imx)
                y_dim = len(imx[0])        
                x_center = int(x_dim/2) 
                y_center = int(y_dim/2)        
                imd = imx[x_center-25:x_center+25, y_center-25:y_center+25]
                
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
    elif curTest == numberTests -1:   
     
            #%%
             
            ## - repeat the code above for all tests and compare iterativally
            ## if it's the last one, procceed. If not, save the above variables and go for the next test 
            
            print("Length for data from tests: " + str(len(data_from_tests)))    
            
            clustering_output = []
            
            M = 0
             
            if len(data_from_tests) > 0:
                
                if curTest == numberTests - 1:   ## for
                
                    couples_combs = []
                    ## numberTests = 10
                    
                    not_include = False  
                    
                    if numberTests == 1:
                        numberTests = 2
                    
                    print("Number tests: ")
                    print(numberTests)
                    
                    # import sys
                    # sys.exit()  
                    
                    x_before = 0
                
                    for x in range(0, numberTests):
                        print("x here: " + str(x))               
                        x_before = x
                        for y in range(0, numberTests):
                            print("y here: " + str(y))                             
                            if x != y:  
                                
                                couple_number_tests = (x,y)           
                                
                                # if len(couples_combs) > 0:
                                #     for x in range(0, numberTests):
                                #         for y in range(0, numberTests):
                                #             couple_x = (x,y)
                                #             if couple_number_tests[0] == couple_x[1] and couple_number_tests[1] == couple_x[0]:
                                #                 not_include = True
                                 
                                if not not_include: 
                                    M += 1 
                                    
                                    print("x: " + str(x))
                                    if isinstance(x, str):
                                        x = x_before 
                                        
                                    print("y: " + str(y))
                                    if isinstance(y, str):
                                        y = int(y)    
                                        
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
                                    
                                    print("OK NOW")
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
                                    
                                    if new_newPathA[1] == 'C' and new_newPathA[0] != 'C':                                        
                                            
                                        new_newPathA = new_newPathA[1:]
                                        
                                        newPathA = new_newPathA
                                    else:
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
                                    
                                    if new_newPathB[1] == 'C' and new_newPathB[0] != 'C':        
                                            
                                        new_newPathB = new_newPathB[1:]
                                        
                                        newPathB = new_newPathB
                                    else:
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
                                    
                                    if roi_bef[1] == 'C' and roi_bef[0] != 'C':        
                                        roi_bef = roi_bef[1:]
                                    
                                    roiPathAfterB = ""
                                    
                                    for p in roi_bef:
                                        if p != '\n' and p != "\n" and p!= "\t" and p != " ":
                                            roiPathAfterB += p
                                    
                                    roi_after = roiPathAfterB
                                    
                                    if roi_after[1] == 'C' and roi_after[0] != 'C':        
                                        roi_after = roi_after[1:]
                                    
                                    print("Complete ROI Bef: " + roi_bef)
                                    print("Complete ROI After: " + roi_after)   
                                     
                                    for i in range(0, count):
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
                                        
                                        if roi_after[1] == 'C' and roi_after[0] != 'C':        
                                        
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
                                                 
                                        
                                        imx = cv2.imread(fixed_roi_path_two + "/roi_image" + str(i) + ".jpg")
                                        print(fixed_roi_path_two + "/roi_image")
                                        print("Image after written")
                                 
                                        cv2.imwrite(roi_after + "/roi_image%d.jpg" % ind_after, imxa)
                                        ind_after += 1                           
                                    
                                    labelsFeatQuality = ['Mean', 'Standard Deviation', 'Mode', 'Median', '1st Quartile', '3rd Quartile', 'Contrast']
                                    
                                    labelsFeatQualityValues = ['Mean', 'Standard Deviation', 'Mode', 'Median', '1st Quartile', '3rd Quartile', 'Contrast']
                                                                     
                                     
                                    # anotherPreviousMetrics = ["Mean", "STD", "Contrast", "ASM", "Max"] 
                                    
                                    # for ind_an, an in enumerate(anotherPreviousMetrics):
                                    #     labelsFeatQualityValues += [str(an)]
                                        
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
                                    
                                    def operations_with_two_matrices(matrix1, matrix2):
                                            """
                                            Perform various operations with two image matrices as input.
                                        
                                            Parameters:
                                            matrix1 (numpy.ndarray): First image matrix.
                                            matrix2 (numpy.ndarray): Second image matrix.
                                        
                                            Returns:
                                            dict: Results of mean, mode, standard deviation, quartiles, and contrast.
                                            """
                                            results = {}
                                            
                                            # Calculate mean
                                            results['mean'] = mean_of_matrices(matrix1, matrix2)
                                            
                                            # Calculate mode
                                            results['mode'] = mode_of_matrices(matrix1, matrix2)
                                            
                                            # Calculate standard deviation
                                            results['std_dev'] = std_dev_of_matrices(matrix1, matrix2)
                                            
                                            # Calculate quartiles
                                            results['quartiles'] = quartiles_of_matrices(matrix1, matrix2)
                                            
                                            # Calculate contrast
                                            results['contrast'] = contrast_of_matrices(matrix1, matrix2)
                                            
                                            return results
                                        
                                    def mean_of_matrices(matrix1, matrix2):
                                            """
                                            Calculate the mean of two image matrices.
                                        
                                            Parameters:
                                            matrix1 (numpy.ndarray): First image matrix.
                                            matrix2 (numpy.ndarray): Second image matrix.
                                        
                                            Returns:
                                            numpy.ndarray: Mean of the two image matrices.
                                            """
                                            return (matrix1 + matrix2) / 2
                                        
                                    def mode_of_matrices(matrix1, matrix2):
                                            """
                                            Calculate the mode of two image matrices.
                                        
                                            Parameters:
                                            matrix1 (numpy.ndarray): First image matrix.
                                            matrix2 (numpy.ndarray): Second image matrix.
                                        
                                            Returns:
                                            int or float or numpy.ndarray: Mode of the two image matrices.
                                            """
                                            combined_data = np.concatenate((matrix1.flatten(), matrix2.flatten()))
                                            from scipy import stats
                                            mode = stats.mode(combined_data)
                                            return mode.mode[0]
                                        
                                    def std_dev_of_matrices(matrix1, matrix2):
                                            """
                                            Calculate the standard deviation of two image matrices.
                                        
                                            Parameters:
                                            matrix1 (numpy.ndarray): First image matrix.
                                            matrix2 (numpy.ndarray): Second image matrix.
                                        
                                            Returns:
                                            float: Standard deviation of the two image matrices.
                                            """
                                            combined_data = np.concatenate((matrix1.flatten(), matrix2.flatten()))
                                            return np.std(combined_data)
                                        
                                    def quartiles_of_matrices(matrix1, matrix2):
                                            """
                                            Calculate the quartiles of two image matrices.
                                        
                                            Parameters:
                                            matrix1 (numpy.ndarray): First image matrix.
                                            matrix2 (numpy.ndarray): Second image matrix.
                                        
                                            Returns:
                                            tuple: First quartile, median, and third quartile of the two image matrices.
                                            """
                                            combined_data = np.concatenate((matrix1.flatten(), matrix2.flatten()))
                                            quartiles = np.percentile(combined_data, [25, 50, 75])
                                            return quartiles
                                        
                                    def contrast_of_matrices(matrix1, matrix2):
                                            """
                                            Calculate the contrast (standard deviation divided by mean) of two image matrices.
                                        
                                            Parameters:
                                            matrix1 (numpy.ndarray): First image matrix.
                                            matrix2 (numpy.ndarray): Second image matrix.
                                        
                                            Returns:
                                            float: Contrast of the two image matrices.
                                            """
                                            mean = mean_of_matrices(matrix1, matrix2)
                                            std_dev = std_dev_of_matrices(matrix1, matrix2)
                                                                    
                                            return std_dev / mean

                                    
                                    folder = fixed_roi_path + '/'
                                    couple_images = []   
                                    
                                    print("See if exists: ")
                                    print(roi_bef + "/roi_image63.jpg")
                                    
                                    # import sys
                                    # sys.exit() 
                                    
                                    for ind in range(0,int(totCountImages)):   ## int(totCountImages/2)
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
                                            
                                    
                                  #  'Mean', 'Standard Deviation', 'Mode', 'Median', '1st Quartile', '3rd Quartile', 'Contrast'
                                  
                                    setMetrics = [] 
                                    setMetricsValues = []
                                    meanList = []
                                    stdList = []
                                    modeList = []
                                    medianList = [] 
                                    firstQuartileList = []
                                    thirdQuartileList = []
                                    contrastList = []
                                    
                                    meanList.insert(0, labelsFeatQualityValues[0])
                                    stdList.insert(0, labelsFeatQualityValues[1])
                                    modeList.insert(0, labelsFeatQualityValues[2])
                                    medianList.insert(0, labelsFeatQualityValues[3])
                                    firstQuartileList.insert(0, labelsFeatQualityValues[4])
                                    thirdQuartileList.insert(0, labelsFeatQualityValues[5])
                                    contrastList.insert(0, labelsFeatQualityValues[6]) 
                                
                                    setMetricsValues.append(labelsFeatQualityValues)
                                    
                                    for ind_couple, couple in enumerate(bigCoupleImages): 
                                        print("Find metrics for couple " + str(ind_couple) + " th couple of images")
                                        origImage = np.array([couple])[0,0,:,:]
                                        deformedImage = np.array([couple])[0,1,:,:]
                                        
                                        results = operations_with_two_matrices(origImage, deformedImage)
                                        
                                        mean = results['mean']
                                        std = results['std_dev']
                                        mode = results['mode']
                                        
                                        print("mean: ")
                                        print(mean)
                                        
                                        print("std: ")
                                        print(std)
                                        
                                        quartiles_info = results['quartiles']                                        
                                        first_quartil = quartiles_info[0]
                                        median = quartiles_info[1]
                                        third_quartil = quartiles_info[2]
                                        
                                        contrast  = results['contrast']
                                        
                                        print("Contrast: ")
                                        print(contrast)
                                        
                                        # import sys
                                        # sys.exit()
                                        
                                        metrics = [mean, std, mode, median, first_quartil, third_quartil, contrast]
                                        
                                        metrics_values = [mean, std, mode, median, first_quartil, third_quartil, contrast]
                                       
                                       
                                        
                                        meanList.append(mean)
                                        stdList.append(std)
                                        modeList.append(mode)
                                        medianList.append(median) 
                                        firstQuartileList.append(first_quartil)
                                        thirdQuartileList.append(third_quartil)
                                        contrastList.append(contrast)
                                       
                                        setMetrics.append(metrics)
                                        setMetricsValues.append(metrics_values)
                                
                                    print("Length for setMetricsValues: " + str(len(setMetricsValues)))
                                        
                                    newListMetrics = []
                                    
                                    # for ind,  metricValue in enumerate(setMetricsValues):
                                    #     number_No_zeros = 0
                                    #     number_No_zeros = np.count_nonzero(np.array([metricValue]))
                                        
                                    #     print("Number no zeros: " + str(number_No_zeros))
                                    #     if int(number_No_zeros) > 6:     ## int(number_No_zeros) > 12
                                    #         newListMetrics.append(metricValue) 
                                    newListMetrics = setMetricsValues
                                    
                                    print("Length for newListMetrics: " + str(len(newListMetrics)))
                                    
                                    metric_1 = []
                                    metric_2 = []
                                    metric_3 = []
                                    metric_4 = []
                                    metric_5 = []
                                    metric_6 = []
                                    metric_7 = []                                   
                                    
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
                                                
                                    listMetricsForEvaluation = [metric_1, metric_2, metric_3, metric_4, metric_5, metric_6, metric_7]
                                      
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
                                    
                                    number_metrics = 7
                                                
                                    if True:
                                        for ind_new_sec, newMetricEval in enumerate(listMetricsForEvaluation):
                                            number_No_zeros = 0
                                            number_No_zeros = np.count_nonzero(np.array([newMetricEval]))
                                            
                                            print("Preliminar metric eval.: " + str(len(newMetricEval)))
                                            
                                            if number_No_zeros > len(newMetricEval)/2:   ## number_No_zeros > len(newMetricEval)/2
                                                secRoundNewListMetrics.append(newMetricEval)
                                            else: 
                                                metricsIndicesDeleted.append(ind_new_sec)
                                   
                                    print("Diff Len: " + str(len(listMetricsForEvaluation) - len(metricsIndicesDeleted)))
                                    
                                    
                                    # import sys
                                    # sys.exit()
                                   
                                    test_size = 0.2
                                    newsecRoundNewListMetrics = []
                                     
                                    lenMax = 0
                                    
                                    if len(secRoundNewListMetrics) > 0:
                                    
                                        for metricSec in secRoundNewListMetrics:
                                            lenMax = len(metricSec)   
                                            
                                        print("len of metricSec: " + str(lenMax))
                                        print("len(secRoundNewListMetrics): " + str(len(secRoundNewListMetrics)))  
                                        
                                        # import sys
                                        # sys.exit()
                                        
                                        # if lenMax%10 != 0:
                                        #    lenMax = round(lenMax/10)*10-10 
                                           
                                        print("len(secRoundNewListMetrics): " + str(lenMax))  
                                         
                                        for metricSec in secRoundNewListMetrics:   
                                            metricSec = metricSec[0:lenMax]
                                            newsecRoundNewListMetrics.append(metricSec) 
                                         
                                        secRoundNewListMetrics = newsecRoundNewListMetrics                        
                                        secRoundNewListMetricsArr = np.array([secRoundNewListMetrics])[0].T
                                        
                                        print("len(secRoundNewListMetricsArr):")
                                        print(len(secRoundNewListMetricsArr))
                                        
                                        # import sys
                                        # sys.exit()
                                    
                                        if int(number_metrics) > len(secRoundNewListMetricsArr[0]):
                                            number_metrics = len(secRoundNewListMetricsArr[0])
                                    else:
                                        print("Not") 
                                        
                               #     print("Number of metrics: " + str(number_metrics))
                                        
                                    resT = np.array([ secRoundNewListMetricsArr[:, int(number_metrics)-1]]).T 
                                    
                                    # print("resT: ")
                                    # print(resT)
                                    
                                    print("secRoundNewListMetricsArr[:, 0:int(number_metrics)-1]: ")
                                    print(secRoundNewListMetricsArr[:, 0:int(number_metrics)-1])
                                    
                                    # import sys
                                    # sys.exit()
                                    
                                    #%%
                                    
                                    # train_data, test_data, labels_train_data, labels_test_data = train_test_split(secRoundNewListMetricsArr[:, 0:int(number_metrics)-1], resT, test_size =test_size, random_state = 42)
                                    # treino_lenght = int((1-test_size)*lenMax)
                                    
                                    # print("train_data")
                                    # print(type(train_data))
                                    # print(len(train_data))
                                    # print(len(train_data[0]))
                                    
                                    # import sys
                                    # sys.exit()
                                    
                                    # Método Silhouette - análise para um nº variável de clusters   
                                    # aux = 0 
                                    # max_silhouette = 0
                                    # silhouette_vector = [] 
                                    # n_clusters3 = range(2, treino_lenght) 
                                    # for j in n_clusters3:
                                    #     km =KMeans(n_clusters=j, max_iter=300, n_init=5).fit(train_data)
                                    #     labels3 = km.labels_
                                    #     silhouette_avg = silhouette_score(train_data, labels3)
                                    #     print("For n_clusters=", j, "The averegae silhouette_score is:", silhouette_avg)   
                                    #     aux = silhouette_avg 
                                    #     if aux > max_silhouette: 
                                    #         max_silhouette = aux
                                    #         number_recommended_clusters = j    
                                    #     silhouette_vector.append(silhouette_avg)  
                                        
                               #     print("\n\nCom base no gráfico Elbow e no método Silhouette, é recomendável formar", number_recommended_clusters, "clusters!") 
                                    
                              #       secRoundNewListMetrics = np.array([secRoundNewListMetrics])[0].T.tolist()
                              #       trainListData = train_data.tolist() 
                              #       nclusters = 6
                                    
                              #       print("trainListData")
                              #       print(type(trainListData))
                              #       print(len(trainListData))
                              #       print(len(trainListData[0]))
                                    
                              #       def pad_sequences(data, max_length, padding_value=0):
                              #           padded_data = []
                              #           for seq in data:
                              #               padded_seq = seq[:max_length] + [padding_value] * (max_length - len(seq))
                              #               padded_data.append(padded_seq)
                              #           return padded_data
                                    
                              #       # Find the maximum length of sequences
                              #       max_length = max(len(seq) for seq in trainListData)

                              #       # Pad sequences to make them uniform in length
                              #       padded_data = pad_sequences(trainListData, max_length)

                              #       # Convert padded data to a 2D NumPy array
                              #       trainListData = np.array(padded_data)
                                     
                              #       kmeans = KMeans(n_clusters=nclusters, max_iter=500, n_init=8).fit(trainListData) 
                              #       Cluster_ID = kmeans.labels_ 
                              #       centroides_A = kmeans.cluster_centers_   
                              # #      print("Centroides dos ",number_recommended_clusters, " clusters recomendados:\n", centroides_A)
                                    
                              #       Cluster_ID_transpose = np.array([np.array([Cluster_ID]).T[0:treino_lenght,0]])
                                    
                              #       objetos_c1 = []
                              #       objetos_c2 = []
                              #       objetos_c3 = [] 
                              #       objetos_c4 = [] 
                              #       objetos_c5 = [] 
                              #       objetos_c6 = [] 
                                    
                              #       for i in range (0, len(Cluster_ID_transpose[0])):
                              #           if Cluster_ID_transpose[0,i] == 0:
                              #               objetos_c1.append(train_data[i, :])
                              #               i_1 = i
                              #           elif Cluster_ID_transpose[0,i] == 1:
                              #               objetos_c2.append(train_data[i, :]) 
                              #               i_2 = i
                              #           elif Cluster_ID_transpose[0,i] == 2:
                              #               objetos_c3.append(train_data[i, :]) 
                              #               i_3 = i
                              #           elif Cluster_ID_transpose[0,i] == 3:
                              #               objetos_c4.append(train_data[i, :]) 
                              #               i_4 = i
                              #           elif Cluster_ID_transpose[0,i] == 4:
                              #               objetos_c5.append(train_data[i, :])
                              #               i_5 = i
                              #           elif Cluster_ID_transpose[0,i] == 5:
                              #               objetos_c6.append(train_data[i, :])  
                              #               i_6 = i     
                                            
                              #       list1 = list(zip(*objetos_c1)) 
                              #       list2 = list(zip(*objetos_c2)) 
                              #       list3 = list(zip(*objetos_c3)) 
                              #       list4 = list(zip(*objetos_c4)) 
                              #       list5 = list(zip(*objetos_c5)) 
                              #       list6 = list(zip(*objetos_c6)) 
                                    
                              #       print(" -- Lists of clusters generated")
                                    
                              #       for l in list1:
                              #           LenList_1 = len(l)
                              #       for l in list2:
                              #           LenList_2 = len(l)
                              #       for l in list3:
                              #           LenList_3 = len(l)
                              #       for l in list4:
                              #           LenList_4 = len(l)
                              #       for l in list5:
                              #           LenList_5 = len(l)
                              #       for l in list6:
                              #           LenList_6 = len(l)    
                                    
                              #       indForFolderClustering = []    
                                    
                              #       list1FirstArr = np.array([np.array([list1[0]])[0]])    
                              #       listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
                                    
                              #       singIndData = []
                                    
                              #       for ind in range(0,len(list1FirstArr[0])):
                              #          tupleIndiceImage = np.where(listArrToCompare == list1FirstArr[0,ind])
                              #          rub, ind_data = tupleIndiceImage
                              #          ind_data = np.array([ind_data])
                                       
                              #          if len(ind_data[0]) == 1:
                              #              singIndData.append(ind_data)
                              #          else:
                              #              if len(ind_data[0]) == 2:
                              #                  print("Not singular")
                              #                  singIndData.append(ind_data[0,0])
                              #                  singIndData.append(ind_data[0,1])           
                                    
                              #       newSingData = []
                                    
                              #       singIndData = np.array([np.unique(np.array([singIndData]))]).tolist()[0]
                                    
                              #       if True:
                              #           for arrSin in singIndData:
                              #               newSingData.append(int(arrSin))
                                    
                              #           newSingIndArr = np.zeros((1,len(newSingData)))     
                              #           newSingIndArr = np.array([newSingData]) 
                                        
                              #           uniqueNewInd = np.array([np.unique(newSingIndArr)])
                                        
                              #           if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
                              #               print("All indices unique")
                                            
                              #               indForFolderClustering.append(uniqueNewInd)    
                                          
                              #       else:
                              #           print("Not equal at phase 2")   
                                    
                              #       list2FirstArr = np.array([np.array([list2[0]])[0]])    
                              #       listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
                                    
                              #       singIndData = []
                                    
                              #       for ind in range(0,len(list2FirstArr[0])):
                              #          tupleIndiceImage = np.where(listArrToCompare == list2FirstArr[0,ind])
                              #          rub, ind_data = tupleIndiceImage
                              #          ind_data = np.array([ind_data])
                                       
                              #          if len(ind_data[0]) == 1:
                              #              singIndData.append(ind_data)
                              #          else:
                              #              print("Not singular")
                              #              singIndData.append(ind_data[0,0])
                              #              singIndData.append(ind_data[0,1])    
                                    
                              #       newSingData = []
                                    
                              #       if True:
                              #           for arrSin in singIndData:
                              #               newSingData.append(int(arrSin))
                                    
                              #           newSingIndArr = np.zeros((1,len(newSingData)))    
                              #           newSingIndArr = np.array([newSingData]) 
                                        
                              #           uniqueNewInd = np.array([np.unique(newSingIndArr)])
                                        
                              #           if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
                              #               print("All indices unique")
                                            
                              #               indForFolderClustering.append(uniqueNewInd) 
                                           
                              #       else:
                              #           print("Not equal at phase 2")   
                                    
                              #       list3FirstArr = np.array([np.array([list3[0]])[0]])    
                              #       listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
                                    
                              #       singIndData = []
                                    
                              #       for ind in range(0,len(list3FirstArr[0])):
                              #          tupleIndiceImage = np.where(listArrToCompare == list3FirstArr[0,ind])
                              #          rub, ind_data = tupleIndiceImage
                              #          ind_data = np.array([ind_data])
                                       
                              #          if len(ind_data[0]) == 1:
                              #              singIndData.append(ind_data)
                              #          else:
                              #              print("Not singular")
                              #              singIndData.append(ind_data[0,0])
                              #              singIndData.append(ind_data[0,1])  
                                    
                              #       newSingData = []
                                    
                              #       if True:
                              #           for arrSin in singIndData:
                              #               newSingData.append(int(arrSin))
                                    
                              #           newSingIndArr = np.zeros((1,len(newSingData)))    
                              #           newSingIndArr = np.array([newSingData]) 
                                        
                              #           uniqueNewInd = np.array([np.unique(newSingIndArr)])
                                        
                              #           if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
                              #               print("All indices unique")
                                            
                              #               indForFolderClustering.append(uniqueNewInd)     
                                          
                              #       else:
                              #           print("Not equal at phase 2")   
                                    
                              #       list4FirstArr = np.array([np.array([list4[0]])[0]])    
                              #       listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
                                    
                              #       singIndData = []
                                    
                              #       for ind in range(0,len(list4FirstArr[0])):
                              #          tupleIndiceImage = np.where(listArrToCompare == list4FirstArr[0,ind])
                              #          rub, ind_data = tupleIndiceImage
                              #          ind_data = np.array([ind_data])
                                       
                              #          if len(ind_data[0]) == 1:
                              #              singIndData.append(ind_data)
                              #          else:
                              #              print("Not singular")
                              #              singIndData.append(ind_data[0,0])
                              #              singIndData.append(ind_data[0,1])  
                                    
                              #       newSingData = []
                                    
                              #       if True:
                              #           for arrSin in singIndData:
                              #               newSingData.append(int(arrSin))
                                    
                              #           newSingIndArr = np.zeros((1,len(newSingData)))    
                              #           newSingIndArr = np.array([newSingData]) 
                                        
                              #           uniqueNewInd = np.array([np.unique(newSingIndArr)])
                                        
                              #           if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
                              #               print("All indices unique")
                                            
                              #               indForFolderClustering.append(uniqueNewInd) 
                                            
                              #       else:
                              #           print("Not equal at phase 2")
                                    
                              #       list5FirstArr = np.array([np.array([list5[0]])[0]])    
                              #       listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
                                    
                              #       singIndData = []
                                     
                              #       for ind in range(0,len(list5FirstArr[0])):
                              #          tupleIndiceImage = np.where(listArrToCompare == list5FirstArr[0,ind])
                              #          rub, ind_data = tupleIndiceImage
                              #          ind_data = np.array([ind_data])
                                       
                              #          if len(ind_data[0]) == 1:
                              #              singIndData.append(ind_data)
                              #          else:
                              #              print("Not singular")
                              #              singIndData.append(ind_data[0,0])
                              #              singIndData.append(ind_data[0,1]) 
                                    
                              #       newSingData = []
                                    
                              #       if True:
                              #           for arrSin in singIndData:
                              #               newSingData.append(int(arrSin))
                                    
                              #           newSingIndArr = np.zeros((1,len(newSingData)))    
                              #           newSingIndArr = np.array([newSingData]) 
                                        
                              #           uniqueNewInd = np.array([np.unique(newSingIndArr)])
                                        
                              #           if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
                              #               print("All indices unique")
                                             
                              #               indForFolderClustering.append(uniqueNewInd) 
                                            
                              #       else:
                              #           print("Not equal at phase 2")
                                        
                              #       list6FirstArr = np.array([np.array([list6[0]])[0]])    
                              #       listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
                                    
                              #       singIndData = []
                                    
                              #       for ind in range(0,len(list6FirstArr[0])):
                              #          tupleIndiceImage = np.where(listArrToCompare == list6FirstArr[0,ind])
                              #          rub, ind_data = tupleIndiceImage
                              #          ind_data = np.array([ind_data])
                                       
                              #          if len(ind_data[0]) == 1:
                              #              singIndData.append(ind_data)
                              #          else:
                              #              print("Not singular")
                              #              singIndData.append(ind_data[0,0])
                              #              singIndData.append(ind_data[0,1]) 
                                    
                              #       newSingData = []
                                    
                              #       if True:
                              #           for arrSin in singIndData:
                              #               newSingData.append(int(arrSin))
                                    
                              #           newSingIndArr = np.zeros((1,len(newSingData)))    
                              #           newSingIndArr = np.array([newSingData]) 
                                        
                              #           uniqueNewInd = np.array([np.unique(newSingIndArr)])
                                        
                              #           if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
                              #               print("All indices unique")
                                             
                              #               indForFolderClustering.append(uniqueNewInd)        
                                           
                              #       else:
                              #           print("Not equal at phase 2")
                                        
                              #       #%%
                                    
                                    
                              #       ###############################################################################################################################################
                              #       ###############################################################################################################################################
                              #       ###############################################################################################################################################
                              #       ###############################################################################################################################################
                              #       ###############################################################################################################################################
                                
                              #       name_folder = first_clustering_storing_output + str(sequence_name) + "_7_8"
                              #       newPath = os.path.join(parent_dir,name_folder)    
                                    
                                    
                              #       if "\n" in newPath or "\t" in newPath or "\r" in newPath or " " in newPath:                                      
                                    
                              #           n_path = ""
                                        
                              #           for let in newPath:
                              #               if let != "\n" and let != "\r" and l != "\t" and l != " ":
                              #                   n_path += let
                                        
                              #           print("NewPath Bef: " + str(n_path))
                              #           n_path = n_path[1:]
                              #           newPath = n_path
                                        
                              #           print("NewPath After: " + str(newPath))
                                        
                              #           if '/' in newPath:
                              #               x = 0
                              #               while os.path.exists(newPath + '/'):
                              #                   newPath += '_' + str(x)
                              #                   x += 1
                              #               nPath = newPath.split('/')
                              #               pas = nPath[-1]
                              #               rpas = nPath[:-1]
                              #               pas = pas[2:]
                              #               pasn = ""
                              #               for p in pas:
                              #                   if p != "\t" and p != " ":
                              #                       pasn += p
                              #               pas = pasn
                              #               rpas.append(pas)
                              #               nnPath = ""
                              #               for r in rpas:
                              #                   nnPath += r + '/'
                                            
                              #               nnPath = nnPath[:-1]
                              #               newPath = nnPath
                                            
                              #           elif "\\" in newPath:
                              #               if os.path.exists(newPath + "\\"):
                              #                   x = 0
                              #                   while os.path.exists(newPath + "\\"):
                              #                       newPath += '_' + str(x)
                              #                       x += 1
                              #                   nPath = newPath.split('/')
                              #                   pas = nPath[-1]
                              #                   rpas = nPath[:-1]
                              #                   pas = pas[2:]
                              #                   pasn = ""
                              #                   for p in pas:
                              #                     if p != "\t" and p != " ":
                              #                        pasn += p
                              #                   pas = pasn
                              #                   rpas.append(pas)
                              #                   nnPath = ""
                              #                   for r in rpas:
                              #                      nnPath += r + '/'
                                                   
                              #                   nnPath = nnPath[:-1] 
                              #                   newPath = nnPath
                                        
                              #           indxp = 0
                              #           for indp, p in enumerate(newPath):
                              #               if indp < 10:
                              #                   if p == "\n" or p == ' ' or p == "\t":
                              #                       indxp = indp
                                                    
                              #           newPath = newPath[indxp+1:]
                                        
                              #           if os.path.exists(newPath + "\\"):
                              #               x = 0
                              #               while os.path.exists(newPath + "\\"):
                                                
                              #                   if newPath[-2] == '_':
                              #                       newPath = newPath[:-2] 
                              #                   if newPath[-3] == '_':
                              #                       newPath = newPath[:-3] 
                                                    
                              #                   newPath += '_' + str(x)
                              #                   x += 1
                              #               else:
                              #                   os.mkdir(newPath) 
                              #       else:
                                        
                              #           if not os.path.exists(newPath + "/"):
                              #               os.mkdir(newPath) 
                              #           else:
                              #               first_new = True
                              #               indNewAdd = 0
                              #               while True:
                              #                   if os.path.exists(newPath + "/"):
                              #                       print("\nYet here " + str(indNewAdd) + " ...")
                              #                       if first_new:
                              #                           newPath = newPath + '_' + str(indNewAdd)
                              #                           indNewAdd += 1
                              #                           first_new = False
                              #                       else:
                              #                           print("Passed it ...")
                              #                           list_underscores = []
                              #                           for indP, p in enumerate(newPath):
                              #                               if p == '_':
                              #                                   list_underscores.append(indP)
                                                                
                              #                           last_pos_underscore = list_underscores[-1]                                                        
                                                        
                              #                           newPathE = newPath[:last_pos_underscore] + '_'
                              #                           newPath = newPathE + str(indNewAdd)
                              #                           indNewAdd += 1
                              #                   else:
                              #                       os.mkdir(newPath)
                              #                       break
                                    
                              #       if not os.path.exists(newPath + "/"):
                              #           os.mkdir(newPath) 
                                     
                              #       print("First Directory created") 
                                    
                              #       trainFolder = "Train_Results" 
                                     
                              #       newPath = os.path.join(newPath + "/", trainFolder)   
                              #       os.mkdir(newPath) 
                                    
                              #       newPath = newPath + "/" 
                                    
                              #       print("Second Directory created")
                                     
                              #       sub_name_folder1 = "Class_1"
                              #       newPath_1 = os.path.join(newPath + "/",sub_name_folder1)
                              #       os.mkdir(newPath_1)
                                     
                              #       sub_name_folder1 = "Class_2"
                              #       newPath_2 = os.path.join(newPath + "/",sub_name_folder1)
                              #       os.mkdir(newPath_2) 
                                         
                              #       sub_name_folder1 = "Class_3"
                              #       newPath_3 = os.path.join(newPath + "/",sub_name_folder1)
                              #       os.mkdir(newPath_3)   
                                     
                              #       sub_name_folder1 = "Class_4" 
                              #       newPath_4 = os.path.join(newPath + "/",sub_name_folder1)
                              #       os.mkdir(newPath_4)
                                     
                              #       sub_name_folder1 = "Class_5"
                              #       newPath_5 = os.path.join(newPath + "/",sub_name_folder1)
                              #       os.mkdir(newPath_5) 
                                         
                              #       sub_name_folder1 = "Class_6"
                              #       newPath_6 = os.path.join(newPath + "/",sub_name_folder1) 
                              #       os.mkdir(newPath_6)   
                                    
                              #       counter_1 = 0
                              #       counter_2 = 0 
                              #       counter_3 = 0
                              #       counter_4 = 0
                              #       counter_5 = 0
                              #       counter_6 = 0
                                    
                                    
                              #       metricsIdTtrain1 = []
                              #       metricsIdTtrain2 = []
                              #       metricsIdTtrain3 = []
                              #       metricsIdTtrain4 = []
                              #       metricsIdTtrain5 = []
                              #       metricsIdTtrain6 = []
                                    
                                    
                              #       print("Cluster_ID_transpose: ")
                              #       print(Cluster_ID_transpose)          
                                    
                              #       for ind_imageInCluster, cluster in enumerate(Cluster_ID_transpose[0]):    ## indForFolderClustering
                              #         ##  cluster_list = cluster.tolist()
                              #        ##   for ind_imageInCluster in cluster_list[0]:
                                    
                              #               image_counter = cv2.imread(roi_bef + "/roi_image" + str(ind_imageInCluster) + ".jpg")
                              #               image_counter_2 = cv2.imread(roi_after + "/roi_image" + str(ind_imageInCluster) + ".jpg")       
                                            
                              #               if cluster == 0:  
                              #                   cv2.imwrite(newPath_1 + "/" + "image_" + str(counter_1) + "_" + str(0) + ".jpg", image_counter) 
                              #                   cv2.imwrite(newPath_1 + "/" + "image_" + str(counter_1) + "_" + str(1) + ".jpg", image_counter_2)
                              #                   metricsIdTtrain1.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                              #                   counter_1 += 1
                              #               if cluster == 1:
                              #                   cv2.imwrite(newPath_2 + "/" + "image_" + str(counter_2) + "_" + str(0) + ".jpg", image_counter)
                              #                   cv2.imwrite(newPath_2 + "/" + "image_" + str(counter_2) + "_" + str(1) + ".jpg", image_counter_2)
                              #                   metricsIdTtrain2.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                              #                   counter_2 += 1
                              #               if cluster == 2: 
                              #                   cv2.imwrite(newPath_3 + "/" + "image_" + str(counter_3) + "_" + str(0) + ".jpg", image_counter)
                              #                   cv2.imwrite(newPath_3 + "/" + "image_" + str(counter_3) + "_" + str(1) + ".jpg", image_counter_2)
                              #                   metricsIdTtrain3.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                              #                   counter_3 += 1 
                              #               if cluster == 3:
                              #                   cv2.imwrite(newPath_4 + "/" + "image_" + str(counter_4) + "_" + str(0) + ".jpg", image_counter)
                              #                   cv2.imwrite(newPath_4 + "/" + "image_" + str(counter_4) + "_" + str(1) + ".jpg", image_counter_2)
                              #                   metricsIdTtrain4.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                              #                   counter_4 += 1
                              #               if cluster == 4:
                              #                   cv2.imwrite(newPath_5 + "/" + "image_" + str(counter_5) + "_" + str(0) + ".jpg", image_counter)
                              #                   cv2.imwrite(newPath_5 + "/" + "image_" + str(counter_5) + "_" + str(1) + ".jpg", image_counter_2)
                              #                   metricsIdTtrain5.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                              #                   counter_5 += 1
                              #               if cluster == 5:
                              #                   cv2.imwrite(newPath_6 + "/" + "image_" + str(counter_6) + "_" + str(0) + ".jpg", image_counter) 
                              #                   cv2.imwrite(newPath_6 + "/" + "image_" + str(counter_6) + "_" + str(1) + ".jpg", image_counter_2)
                              #                   metricsIdTtrain6.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                              #                   counter_6 += 1
                                    
                              #       LenListsClusters1= [LenList_1, LenList_2, LenList_3, LenList_4, LenList_5, LenList_6]
                                    
                              #       ##########################################################################################################
                              #       ##########################################################################################################
                              #       ##########################################################################################################
                              #       ##########################################################################################################
                                    
                              #       m_1_f = []
                              #       m_2_f = []
                              #       m_3_f = []
                              #       m_4_f = []
                              #       m_5_f = []
                              #       m_6_f = []
                                    
                                    
                              #       metrics_afterClustering = []
                                    
                              #       if len(metricsIdTtrain1) == 0: 
                              #           print("Cluster not formed. Discarding this one")
                              #       else:
                              #           metricsIdTrain1 = np.array([metricsIdTtrain1]).T.tolist()
                              #           metrics_afterClustering.append(metricsIdTrain1) 
                                        
                              #           for m_1 in metricsIdTrain1:
                              #               m_1_f.append(np.mean(np.array([m_1])[0,:,0]))    
                                        
                              #       if len(metricsIdTtrain2) == 0: 
                              #           print("Cluster not formed. Discarding this one")
                              #       else:
                              #           metricsIdTrain2 = np.array([metricsIdTtrain2]).T.tolist()
                              #           metrics_afterClustering.append(metricsIdTrain2)
                                        
                              #           for m_2 in metricsIdTrain2:
                              #               m_2_f.append(np.mean(np.array([m_2])[0,:,0]))
                                        
                              #       if len(metricsIdTtrain3) == 0: 
                              #           print("Cluster not formed. Discarding this one")
                              #       else:
                              #           metricsIdTrain3 = np.array([metricsIdTtrain3]).T.tolist()
                              #           metrics_afterClustering.append(metricsIdTrain3)
                                        
                              #           for m_3 in metricsIdTrain3:
                              #               m_3_f.append(np.mean(np.array([m_3])[0,:,0]))
                                        
                              #       if len(metricsIdTtrain4) == 0: 
                              #           print("Cluster not formed. Discarding this one")
                              #       else:
                              #           metricsIdTrain4 = np.array([metricsIdTtrain4]).T.tolist()
                              #           metrics_afterClustering.append(metricsIdTrain4)
                                        
                              #           for m_4 in metricsIdTrain4:
                              #               m_4_f.append(np.mean(np.array([m_4])[0,:,0]))
                                        
                              #       if len(metricsIdTtrain5) == 0: 
                              #           print("Cluster not formed. Discarding this one")
                              #       else:
                              #           metricsIdTrain5 = np.array([metricsIdTtrain5]).T.tolist()
                              #           metrics_afterClustering.append(metricsIdTrain5)
                                        
                              #           for m_5 in metricsIdTrain5:
                              #               m_5_f.append(np.mean(np.array([m_5])[0,:,0]))
                                        
                              #       if len(metricsIdTtrain6) == 0: 
                              #           print("Cluster not formed. Discarding this one")
                              #       else:
                              #           metricsIdTrain6 = np.array([metricsIdTtrain6]).T.tolist()
                              #           metrics_afterClustering.append(metricsIdTrain6)
                                        
                              #           for m_6 in metricsIdTrain6:
                              #               m_6_f.append(np.mean(np.array([m_6])[0,:,0]))
                                            
                                    
                              #       print("metrics_afterClustering:")
                              #       print(metrics_afterClustering)
                                    
                              #       print("secRoundNewListMetricsArr: ")
                              #       print(secRoundNewListMetricsArr)
                              #       #################################################################################################################
                              #       #################################################################################################################
                              #       #%%
                                    
                                  
                                    labelsMetricsToScore = ['Mean', 'Standard Deviation', 'Mode', 'Median', '1st Quartile', '3rd Quartile', 'Contrast']
                                    
                                    stdListValues = []
                                    global_std = [] 
                                    
                                    metrics_afterClustering = secRoundNewListMetricsArr
                                    
                            #        print("metrics_afterClustering: ") 
                            #        print(metrics_afterClustering)
                                    
                                    secondMetricTable = metrics_afterClustering.T.tolist() 
                                    
                                    print("secondMetricTable: ")
                                    print(len(secondMetricTable))
                                    print(len(secondMetricTable[0]))
                                    print(len(secondMetricTable[0][0]))
                                    print(len(secondMetricTable[0][0][0]))
                                    
                                    # import sys
                                    # sys.exit()  
                                    
                                    listOfFlatten = []
                                    stdListValuesMetrics = [] 
                                    
                                    if len(secondMetricTable[0][0][0]) == 1:
                                    
                                        for indSec, secInd in enumerate(secondMetricTable):    
                                            flattenSecond = []
                                            
                                            flatten_list = []
                                            
                                            for sublist in secInd:
                                                print(sublist)
                                                if isinstance(sublist, list):
                                                    for element in sublist:                                                
                                                        flatten_list.append(element) 
                                            
                                            for fla in flatten_list:
                                                fla_one = np.array([fla])[0]  ## [0,0]
                                                flattenSecond.append(fla_one)
                                        
                                            listOfFlatten.append(flattenSecond)
                                    else:
                                        
                                        for indSec, secInd in enumerate(secondMetricTable):    
                                            flattenSecond = []
                                            
                                            flatten_list = []
                                            
                                            for sublist in secInd:
                                                if isinstance(sublist, list):
                                                    for element in sublist:
                                                        for px in element:
                                                            flatten_list.append(px)                                                
                                                        
                                            for fla in flatten_list:
                                                fla_one = np.array([fla])[0]  ## [0,0]
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
                                        if len(indS) > 0:
                                            indS = np.array([indS])[0]
                                            if len(indS[0]) > 0:
                                                indS = np.array([indS])[0,0]
                                            indOfSorted = indS
                                            sortedIndices.append(indOfSorted)
                                    
                                    sortedIndicesToGo = sortedIndices[0:15]
                                    
                                    exlistToPCA = secRoundNewListMetricsArr.T.tolist()
                                    
                                    print("exlistToPCA: ")
                                    print(exlistToPCA)
                                    
                                    # import sys
                                    # sys.exit()                                    
                                    
                                    for ind_listPCA, listPCA in enumerate(exlistToPCA):
                                        if ind_listPCA in sortedIndicesToGo:   ##  and ind_listPCA != 16
                                            listToPCA.append(listPCA) 
                                    
                                    if len(listToPCA) == 0:
                                        listToPCA = exlistToPCA
                                            
                                    filtered_metrics = []
                                    newstdlistValuesMetrics_sec = []
                                    metricsToPCA_analysis = [] 
                                    metricsToPCA_norm = []
                                    
                                    # for ind_mat1, mat1 in enumerate(metrics_afterClustering):
                                    #     mat1_n = []
                                    #     for ind_mat2, mat2 in enumerate(mat1):
                                    #         mat2_n = [] 
                                    #         for ind_mat3, mat3 in enumerate(mat2):
                                    #             mat3 = mat3[0]
                                    #             mat2_n.append(mat3)
                                    #         mat1_n.append(mat2_n)  
                                        
                                    #     metricsToPCA_analysis.append(mat1_n)
                                     
                                    ## Standard normalization ########################################################################################################
                                    ##################################################################################################################################
                                    
                                    metricsToPCA_norm = []
                                    
                                    print("listToPCA: ")
                                    print(listToPCA)
                                    
                                    # Write the NumPy array to the text file
                              ##      np.savetxt("listtopca_var.txt", listToPCA)
                                    
                                    for met1 in listToPCA: 
                                        
                                            print("met:")
                                            print(met1)                                   
                                        
                                            mean_value = np.mean(np.array([met1]))
                                            std_value = np.std(np.array([met1])) 
                                            
                                            print("std_value:")
                                            print(std_value) 
                                            
                                            # import sys
                                            # sys.exit()
                                            
                                            
                                            metricPCA_norm1 = [] 
                                            
                                            for met2 in met1:
                                                print("met2: ")
                                                print(met2)
                                                
                                                if isinstance(met2, list):
                                                    # Flatten the matrix to a single list
                                                    flat_list = [item for sublist in met2 for item in sublist]
                                                    
                                                    # Calculate the mean of the values in the flat list
                                                    met2x = sum(flat_list) / len(flat_list)
                                                else:
                                                    met2x = np.mean(met2)
                                                
                                                met2 = met2x
                                                    
                                                if std_value != 0:
                                                    metricPCA_norm1.append((met2-mean_value)/std_value)        
                                                else:
                                                    metricPCA_norm1.append(0)
                                            
                                                # for m in metricPCA_norm1:
                                                #     if np.isnan(m):
                                                #         print("Here - nan value")
                                                #         print("mean_value: " + str(mean_value))
                                                        
                                            metricsToPCA_norm.append(metricPCA_norm1)
                                            
                                            
                                            
                                            # import sys
                                            # sys.exit()
                                        
                                    
                                    print("MetricsToPCA_norm: ")
                                    print(metricsToPCA_norm)                                    
                                  
                                    ##################################################################################################################################
                                    #%%
                                    
                                    # nan_count = np.isnan(metricsToPCA_norm).sum()
   
                                  
                                    # if nan_count > len(metricsToPCA_norm) / 2:
                                    #    metricsToPCA_norm = listToPCA
                                    # else: 
                                    metricsToPCA_norm = [list(row) for row in zip(*metricsToPCA_norm)]
                                    
                                    print("MetricsToPCA_norm: ")
                                    print(metricsToPCA_norm)
                                    
                                    # import sys
                                    # sys.exit()
                                    
                                    dfxi = pd.DataFrame(data=metricsToPCA_norm)  
                                    
                                    print("dfxi before: ")
                                    print(dfxi)
                                    
                                    dfxi = dfxi.dropna(axis=1,how='all')
                                    
                                    print("dfxi here 1: ")
                                    print(dfxi)                                     
                               
                                    # import sys
                                    # sys.exit()
                                    
                                    # print("Dataframe: ") 
                                    # print(dfxi.to_string())   
                                    
                                    pcai = PCA(n_components=None)                             
                                    
                                    dfx_pcai = pcai.fit(dfxi)      ## Error
                                     
                                    X_pcai = pcai.transform(dfxi)   
                                    dfx_transi = pd.DataFrame(data=X_pcai)
                                    
                                    ## secRoundNewListMetricsArr
                                    
                                    plt.figure()
                                    plt.scatter(dfx_transi[0], dfx_transi[1], c ="blue")
                                    plt.title("Correlation between first two PCA components - [" + str(x) + "," + str(y) + "]")
                                    plt.xlabel("First PCA component")
                                    plt.ylabel("Second PCA component")                                   
                                    thisDir = os.getcwd()
                                    dirResultsOutput = thisDir + '\\GraphsOutput\\'
                              #      plt.savefig(dirResultsOutput + "corr_pca_" + str(x) + "_" + str(y) + ".png")
                                    plt.show() 
                                    
                                    print("x: " + str(x))
                                    
                                    print("y: " + str(y))
                                    
                                    print("Number tests: ")
                                    print(numberTests)      
                                    
                                    top_four_indices = sorted(range(len(pcai.explained_variance_)), key=lambda i: pcai.explained_variance_[i], reverse=True)[:4]

                                    # Get the labels of the top 4 features
                                    remainingMetricsToClustering = [labelsMetricsToScore[i] for i in top_four_indices]
                                    
                                    # Select only the top 4 principal components
                                    dfx_top_four_pcai = X_pcai[:, top_four_indices]
                                    
                                    # Convert the resulting DataFrame to a list of lists
                                    trainListData = dfx_top_four_pcai.tolist()
                                     
                                    # pca_coef_feat_first_comp = dfx_transi[0].tolist()
                                    # abs_pca_coeff = np.array([abs(np.array([pca_coef_feat_first_comp])[0])])
                                    # abs_pca_coeffList = abs_pca_coeff.tolist()
                                    
                                    # print("abs_pca_coeffList:")
                                    # print(abs_pca_coeffList)
                                    
                                    # ## sortedPCA_coeff = sorted(abs_pca_coeffList[0], reverse=True)
                                    # # Sort the absolute values of the PCA coefficients
                                    # sortedPCA_coeff = sorted(abs_pca_coeffList[0])
                                    
                                    # # Extract the top 4 absolute values
                                    # top_four = sortedPCA_coeff[-4:]
                                     
                                    # print("top_four: ")
                                    # print(top_four)
                                    
                                    # # Find the indices of the top 4 absolute values in the original list
                                    # sortedIndices2 = [abs_pca_coeffList[0].index(value) for value in top_four]
                                    
                                    # print("sortedIndices2: ")
                                    # print(sortedIndices2)
                                    
                                    # # Identify double sorted elements and select only one of them
                                    # indicesSortedFromPCA = []
                                    # for sorted_ind in sortedIndices2:
                                    #     if sorted_ind not in indicesSortedFromPCA:
                                    #         indicesSortedFromPCA.append(sorted_ind)
                                    #     else:
                                    #         # In case of duplicates, choose not to add to the list
                                    #         pass
          
                                    # remainingMetricsToClustering = [] 
                              #      trainListData = []
                                    
                                    # print("indicesSortedFromPCA: ")
                                    # print(indicesSortedFromPCA)
                                    
                                    # print("len(exlistToPCA):")
                                    # print(len(exlistToPCA))
                                    
                                    # for ind in indicesSortedFromPCA: 
                                    #     if ind <= 21 and ind < len(exlistToPCA):
                                    #         remainingMetricsToClustering.append(labelsMetricsToScore[ind])   
                                    #         trainListData.append(exlistToPCA[ind]) 
                                             
                                    nclusters = 2
                                      
                              #      trainListData = np.array([trainListData])[0].T.tolist()
                                    
                                    train_data = np.array([trainListData])[0].T
                                    treino_lenght = 370
                                    
                                    print("trainListData: ")
                                    print(trainListData)
                                     
                                    kmeans = KMeans(n_clusters=nclusters, max_iter=500, n_init=8).fit(trainListData) 
                                    Cluster_ID = kmeans.labels_ 
                                    centroides_A = kmeans.cluster_centers_   
                              ##      print("Centroides dos ",number_recommended_clusters, " clusters recomendados:\n", centroides_A)
                                    
                                    print("Cluster_ID")
                                    print(Cluster_ID)
                                    
                                    Cluster_ID_transpose = np.array([np.array([Cluster_ID]).T[0:treino_lenght,0]])
                                    
                                    print("Cluster_ID_transpose")
                                    print(Cluster_ID_transpose)
                                    
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
                                    
                                    print("classAFurther: ")
                                    print(classAFurther)
                                    
                                    print("classBFurther: ")
                                    print(classBFurther)
                                                
                                    number_recommended_clusters = nclusters
                                                
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
                                    dirResultsOutput = thisDir + '\\GraphsOutput\\Results\\'
                                    
                                    if os.path.isdir(dirResultsOutput) == False:  
                                    
                                        dirResults = os.path.join(dirResultsOutput) 
                                        os.mkdir(dirResults)    
                                    
                                    #####
                                    
                                    print("dfx_transi1: ")
                                    print(dfx_transi1)
                                    
                                    print("dfx_transi2: ")
                                    print(dfx_transi2) 
                                    
                                    if len(dfx_transi1[0]) > 0 and len(dfx_transi1[1]) > 0:    
                                        plt1 = plt.scatter(dfx_transi1[0], dfx_transi1[1], c ="blue")
                                    if len(dfx_transi2[0]) > 0 and len(dfx_transi2[1]) > 0:
                                        plt2 = plt.scatter(dfx_transi2[0], dfx_transi2[1], c ="red")
                                        
                                    plt.legend((plt1, plt2),
                                                ('Class A', 'Class B'))
                                    plt.title("Correlation between first two PCA components")
                                    plt.xlabel("First PCA component")
                                    plt.ylabel("Second PCA component")    
                                    
                                   
                                    plt.savefig(dirResultsOutput + "pca_graph_" + str(x) + "_" + str(y) + ".png")
                                    
                                    plt.show()  
                                    
                                    data_results.append(clustering_inf_data)
                                    
                                    data_results.append(dirResultsOutput + "pca_graph_" + str(x) + "_" + str(y) + ".png")
                                    
                                    
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
                                    plt.savefig(dirResultsOutput + "distances_firstCluster_" + str(x) + "_" + str(y) + ".png")
                                    plt.show()
                                    
                                    data_results.append(dirResultsOutput + "distances_firstCluster_" + str(x) + "_" + str(y) + ".png")
                                    
                                    plt.scatter(xDist2, dists2, c ="blue")
                                    plt.title("Distance to centroid B")
                                    plt.xlabel("Number of point")
                                    plt.ylabel("Distance of points of second cluster to its centroid")
                                    plt.savefig(dirResultsOutput + "distances_secondCluster_" + str(x) + "_" + str(y) + ".png")  
                                    plt.show() 
                                    
                                    data_results.append(dirResultsOutput + "distances_secondCluster_" + str(x) + "_" + str(y) + ".png")
                                    
                                        
                                    print(" -- Lists of clusters generated")
                                    
                                    for l in list1:
                                        LenList_1 = len(l)
                                    for l in list2:
                                        LenList_2 = len(l)
                                    
                                #     #%%
                                        
                                #     trainListData = np.array([trainListData])[0].T.tolist()   
                                        
                                #     Y_euclidean = pdist(trainListData, metric='euclidean')
                                #     Y_euclidean_square = squareform(Y_euclidean)
                                #     Y_cityblock = pdist(trainListData, metric='cityblock')
                                #     Y_euclidean_square = squareform(Y_cityblock)
                                    
                                #     Z_euclidean_average = linkage(trainListData, method='average', metric='euclidean')
                                #     Z_euclidean_ward = linkage(trainListData, method='ward', metric='euclidean')
                                    
                                #     Z_cityblock_average = linkage(trainListData, method='average', metric='cityblock') 
                                    
                                #     distances_from_euclidean_average = Z_euclidean_average[:,2].tolist()
                                    
                                #     clustersList = Z_euclidean_average[:,0].tolist() + Z_euclidean_average[:,1].tolist()
                                    
                                #     totNumbClusters_firstExp = np.max(np.array([clustersList])[0])
                                    
                                #     numberObservationsForEachCluster_first = Z_euclidean_average[:,3].tolist()
                                #     totObservationsFirst = np.sum(np.array([numberObservationsForEachCluster_first]))
                                    
                                    
                                #     distances_from_euclidean_ward = Z_euclidean_ward[:,2].tolist()
                                    
                                #     clustersList = Z_euclidean_ward[:,0].tolist() + Z_euclidean_average[:,1].tolist()
                                    
                                #     totNumbClusters_secExp = np.max(np.array([clustersList])[0])
                                    
                                #     numberObservationsForEachCluster_second = Z_euclidean_ward[:,3].tolist()
                                #     totObservationsSec = np.sum(np.array([numberObservationsForEachCluster_second]))
                                    
                                    
                                    
                                #     distances_from_cityblock_average = Z_cityblock_average[:,2].tolist()
                                    
                                #     clustersList = Z_cityblock_average[:,0].tolist() + Z_euclidean_average[:,1].tolist()
                                    
                                #     totNumbClusters_thirdExp = np.max(np.array([clustersList])[0])
                                    
                                #     numberObservationsForEachCluster_third = Z_cityblock_average[:,3].tolist()
                                #     totObservationsThird = np.sum(np.array([numberObservationsForEachCluster_third]))
                                    
                                #     distancesMeanVar = [np.mean(np.array([distances_from_euclidean_average])), np.mean(np.array([distances_from_euclidean_ward])), np.mean(np.array([distances_from_cityblock_average]))]
                                #     labelsMeasuresDistances = ['Euclidean Average', 'Euclidean Ward', 'Cityblock Average']
                                    
                                #     maxMeanDistance = 0
                                    
                                #     for indDist, distMeanValue in enumerate(distancesMeanVar):
                                #         if distMeanValue > maxMeanDistance:
                                #             maxMeanDistance = distMeanValue
                                #             indMaxMeasureDistance = indDist 
                                    
                                #     print("Selected Measure for distance between clusters: " + labelsMeasuresDistances[indMaxMeasureDistance])
                                    
                                #     distancesBetClustersBestMeasure = []
                                    
                                #     if indMaxMeasureDistance == 0:
                                #         distancesBetClustersBestMeasure = Z_euclidean_average.tolist()
                                #         numberClustersFromDist = totNumbClusters_firstExp
                                #     else:
                                #         if indMaxMeasureDistance == 1:
                                #             distancesBetClustersBestMeasure = Z_euclidean_ward.tolist()
                                #             numberClustersFromDist = totNumbClusters_secExp
                                #         else:
                                #             if indMaxMeasureDistance == 2:
                                #                 distancesBetClustersBestMeasure = Z_cityblock_average.tolist()
                                #                 numberClustersFromDist = totNumbClusters_thirdExp
                                                
                                #     distance_output = []
                                                
                                #     for dist1 in distancesBetClustersBestMeasure:
                                #         distance_output2 = []
                                #         for ind_dist2, dist2 in enumerate(dist1):
                                #             if ind_dist2 != 2:
                                #                 distance_output2.append(int(dist2))
                                #             else:
                                #                 distance_output2.append(dist2)
                                #         distance_output.append(distance_output2)
                                        
                                #     #### Comparison of number of clusters between distances approach and the clustering one:
                                #     if  number_recommended_clusters == numberClustersFromDist:
                                #         print("The above methods provide the same number of clusters")
                                #     else:
                                #         if number_recommended_clusters > numberClustersFromDist:
                                #             print("The number of recommended clusters (from clustering approach) is higher than the number of clusters computed from dist-linkage method.")
                                #         else:
                                #             if number_recommended_clusters < numberClustersFromDist:
                                #                 print("The number of clusters computed from dist-linkage method is higher than the number of recommended clusters (from clustering approach).")
                                    
                                    
                                #     #%%     
                                    
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
                                    
                                #     #%%
                                    
                                    k1 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
                                    k2 = [[1, 0, -1],  [2, 0, -2], [1, 0, -1]]
                                    
                                    k1_arr = np.array([k1])[0]
                                    k2_arr = np.array([k2])[0]
                                    
                                    M_c = []
                                     
                                    #%%
                                    
                                    ind_slash = 0
                                    
                                    for ind_x, x in enumerate(IFVP):
                                        if x == '/':
                                            ind_slash = ind_x
                                            break
                                    
                                    IFVP_p = IFVP[ind_slash:]
                                    IFVP = 'C:' + IFVP_p
                                            
                                    
                                    if IFVP[0] != 'C' and IFVP[0] == ':':
                                        IFVP = 'C' + IFVP
                                    elif IFVP[0] != 'C' and  IFVP[0] != ':':
                                        IFVP = IFVP[1:]
                                    
                                    print(IFVP + str(sequence_name) + "_2" + "/video_image")
                                    
                                    for video_image in range(0,count):           #### 321   ## from 361   ## to count = 637
                                        
                                            print("Analysing for image " + str(video_image) + " th")
                                        
                                            image = cv2.imread(IFVP + str(sequence_name) + "_2" + "/video_image%d.jpg" % video_image)        
                                            
                                            if image is not None:
                                                imx = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
                                               
                                           
                                             
                                                M1 = convolve2D(imx, k1)
                                                M2 = convolve2D(imx, k2)      
                                                
                                                
                                                print("Generating output for image " + str(video_image))
                                                
                                                comp_x = np.power(np.array([M1])[0], 2)      
                                                comp_y = np.power(np.array([M2])[0], 2) 
                                                
                                                sum_comps = comp_x + comp_y 
                                                
                                                gen_output = np.power(sum_comps, 1/2).astype(int)
                                                
                                                cv2.imwrite(roi_path + "/gen_output_%d.jpg" % video_image, gen_output)
                                                
                                        #######################################
                                        #######################################
                                        #######################################
                                        #######################################
                                        #######################################  
                                         
                                                executionTime = (time.time() - startTime)
                                                print('Execution time in seconds: ' + str(executionTime))
                                            else:
                                                break
                                        
                                    clustering_output.append((clustering_inf_data, executionTime, totCountImages))
                                     
                                    
                                else:
                                   not_include = True     
             
         
             
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
    
    
       
##    return percIntervalForK, executionTime, totCountImages
            # import sys
            # sys.exit()
        

            print("M: " + str(M))                   
                
            if len(clustering_output) == 1:
                  clustering_output = clustering_output[0]
                    
            print("Returning clustering output")
            print(clustering_output)
                    
            time.sleep(30)
            return clustering_output   
 
 