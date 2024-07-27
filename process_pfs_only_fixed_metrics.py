# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:13:51 2023

@author: Rui Pinto
"""

import PySimpleGUI as sg
from processingScript_until_features_fixed_f import videoAnalysisJustFeatures, videoAnalysisMultipleVideosWithinOneJustFeatures
from soft_single_test import videoAnalysis_single_justFeatures
from soft_single_test import videoAnalysis_single
from countdown_timer import countdown_timer_display 
from info_gui import theory_info
from txt_files_searcher import proc_txt_browsing
from interactive_approach__withinVideo import sepVideosFromWholeOne
from texts_info import read_from_txt_file_tests_info, read_configs_file
from features_pan_auto_functions import features_auto
import time
import webbrowser  
import shlex
import subprocess
import os 
import io   
import cv2 
import numpy as np   
from PIL import Image
import pandas as pd 
import sys

def read_number_tests_from_file():
    with open('temp_numberTests.txt', 'r') as file:
        numberTests = int(file.read())
 
    # Delete file after reading
    import os
    os.remove('temp_numberTests.txt')

    return numberTests 


def load_packages():
    import PySimpleGUIWeb as sg
    import PySimpleGUI as sg_py 
    from pypylon import pylon   
    import numpy as np 
    import io   
    import os   
    from PIL import Image 
    ## from paramsPylon import setParametersToPypylon 
    from full_code import whole_processing_software, whole_processing_software_adv
    from open_form_gui import exp_control 
    from countdown_timer import countdown_timer_display 
    from info_gui import theory_info 
    from pfs_input import read_pfs_file 
    # from pfs_input_acq_step_only import read_pfs_file
    from getCurrentDateAndTime import getDateTimeStrMarker 
    import keyboard  
    import time  
    import webbrowser  
    from fac_params import extra_params_gui 
    from gui_advanced import adv_params_gui 
    from softwareImageProcessingForLaserSpeckleAnalysisFinal import videoAnalysis
    from check_basler_camera import confirm_basler
    from live_camera_image import acq_image_camera 
    from check_centered_image import help_camera_center, get_ok, set_ok
    from test_image_preview import get_test_image
    from optional_extra_prop import optional_prop, optional_prop_norm  
    from gen_gui_working import control_gui
    from common import splitfn  
    import threading 
    ## from threading import Event 
    import multiprocessing  
    from threading import Thread 
    import glob
    import cv2 
    import subprocess 
    import functools    
    import shlex  
    import imageio   
    import sys
    
def listFeatures(list_features):
    
    print("List Features function")
    df = open('list_features.txt', 'w') 
    
    df.write("List of features: ")
    df.write('\n\n')
    for d in list_features:
        df.write(d)
        df.write('\n')
        
    df.close()
    
    contents = open('list_features.txt', 'r')
    
    with open("list_features.html", "w") as e:
        for lines in contents.readlines():
            
            lines = lines[:-1]   
            
            if not ("List of features:" in lines):
                lines = lines.replace(" ", "")
            
            print(lines)
    
            e.write(lines + "<br>\n")    ## <br>           
            
    webbrowser.open('list_features.html', new=2)   


def save_to_excel(file_name, data, metrics_names):
    # Add a column with image numbers (0 to 50)
    image_numbers = np.arange(0, len(data)).reshape(-1, 1)
    data_with_images = np.hstack((image_numbers, data))

    # Add a row with feature numbers (1 to 7)
    feature_numbers = np.arange(1, 8).reshape(1, -1).tolist()[0]
    
    if metrics_names == None:      
        
        feature_numbersx = [] 
        feature_numbersx.append('x')
    
        for i in feature_numbers:
            feature_numbersx.append(str(i))
        
        data_with_images_and_features = np.vstack((feature_numbersx, data_with_images))
        
    else: 
        feature_numbers = metrics_names
    
        feature_numbersx = [] 
        feature_numbersx.append('x')
    
        for i in feature_numbers:
            feature_numbersx.append(str(i))
        
        print("Actual features \n ")
        print("Features numbers: ")
        print(feature_numbersx)
        print("Data w images: ") 
        print(data_with_images) 
        
        data_with_images1 = data_with_images.tolist()
        
        data_with_images_and_features = []

        data_with_images_and_features.append(feature_numbersx)

        for f in data_with_images1:
            data_with_images_and_features.append(f)     
       
  
        print("data_with_images_and_features: ")
        print(data_with_images_and_features)
 
    # Convert to a pandas DataFrame
    df = pd.DataFrame(data_with_images_and_features)
    
 #   writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
 
    # Write to Excel file
 #   df.to_excel(file_name, index=False, header=False, sheet_name='Sheet1')
 
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

    # Write the DataFrame to Excel
    df.to_excel(writer, index=False, header=False, sheet_name='Sheet1')
    
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']    
   
     
    bold_format = workbook.add_format({'bold':True})
    
    worksheet.set_row(0, cell_format=bold_format)
    worksheet.set_column('A:A', cell_format=bold_format)
    
    writer.save()   

### Reduzir o tamanho da janela e ver possiblidade de ajustar com o rato para ficar maior ou menor (autoajuste)
def gui_show_results(featuresData, metrics_names):
    
    print("Saving features")   
   
    list_listsF = []
    featuresData_sec = featuresData[0]
    list_f, number_f = featuresData_sec
    
    for ind_f, f in enumerate(list_f):
       
        # print("For indice " + str(ind_f) + " : list is ")
        # print(list_f)
        # print("\n")
        
        list_listsF.append(list_f)
    
    min_number_features = number_f
    
    print("min_number_features: " + str(min_number_features))
    
    if min_number_features > 1:
       
        print("Ok")
        # newListListF = []
        
        # for l in list_listsF:
        #     l_now = l[:min_number_features]
        #     newListListF.append(l_now)
        
        # print("Recalculating list of features ...")
            
        # for indListF, listF in enumerate(newListListF):
        #     print("For indice " + str(indListF) + " : ")
        #     print(listF) 
        #     print("\n")
    else:
        print("Check features")
        raise Exception 
     
    ## Write to an excel file info with:
    ## process number ## features
    ## GUI to require name of excel file
    
    print("Now A")

    layout = [
        [sg.Text("Enter the Excel file name (without extension):")],
        [sg.InputText(), sg.Button("Save")],
    ]

    window = sg.Window("Save to Excel", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break

        if event == "Save":
            print("Now B")
            
            file_name = values[0] + ".xlsx"
            list_f_T = list_f   ## .T
            print("Excel filename to save the features data: " + str(file_name))
     #       data = np.random.rand(7, 50)  # Replace this with your 2D numpy array
            data = list_f_T
            print("Data being written: ")
            print(data)
            print("\n\n")
            
            save_to_excel(file_name, data, metrics_names)
            sg.popup(f"Data saved to {file_name}")
            
            time.sleep(5)            
            break

    window.close()           
          
         
def post_proc_pfs_only_fixed_metrics():
    
    curTest = 0    
    time_bet_tests = 0 
    completed = False
    
    this_dir = os.path.abspath(__file__)
    
    dParts = this_dir.split("\\")

    newDir = ""
    for indD, d in enumerate(dParts):
        if indD < len(dParts)-1:
            newDir += d + "\\"
    this_dir =  newDir      
    
    dir_txt_file, config_dirs = proc_txt_browsing(this_dir)
    
    print("pppppppppppppppppp")
    print(dir_txt_file)
    print(config_dirs)
    print("pppppppppppppppppp")
    
    infx = [] 
    
    if True:         
         
        again = True
        
        if curTest == 0: 
            
            
            dir_tests_info_path = dir_txt_file
            
            # while again == True:
            
            #     windowx = sg.Window('Choose path to tests info file', [[sg.Text('File name')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
            #     (keyword, dict_dir) = windowx                
            
            #     dir_tests_info_path = dict_dir['Browse'] 
                
            #     if dir_tests_info_path is None:
            #         again = True
            #     else:
            #         if not ".txt" in dir_tests_info_path:
            #             again = True
            #         else:
            #             again = False
            #             break
            
            data_tests = read_from_txt_file_tests_info(dir_tests_info_path)
             
            [numberTests, dur_test, time_bet_tests] = data_tests 
            
        print("Number of tests: " + str(numberTests))
        print("Duration of each test: " + str(dur_test))
        print("Time between tests: " + str(time_bet_tests))
        
        time_bet_tests = round(float(time_bet_tests/60),5)
    
        print("Test number " + str(curTest+1))   
        
        load_packages()        
      
        
        
        if numberTests == 1:
              data_to_save = [] 
              
              if True:      
                  
                  again = True
                  
                  dir_configs_path = config_dirs[0]
                  
                  # while again == True: 
                  
                  #     windowx = sg.Window('Choose path to configs file', [[sg.Text('File name')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
                  #     (keyword, dict_dir) = windowx                
                  
                  #     dir_configs_path = dict_dir['Browse'] 
                      
                  #     if dir_configs_path is None:
                  #         again = True
                  #     else:
                  #         if not ".txt" in dir_configs_path:
                  #             again = True
                  #         else:
                  #             again = False
                  #             break
                  
                  print(dir_configs_path)
                   
                  configs_list = read_configs_file(dir_configs_path)    
                  
                  again = True
                  
                  pathRoiStart = ""
                  pathRoiEnd = ""
                  
                  while again == True:
                  
                      windowx = sg.Window('Choose path for images before', [[sg.Text('Folder name')], [sg.Input(), sg.FolderBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
                      (keyword, dict_dir) = windowx                
                   
                      dir_bef_path = dict_dir['Browse'] 
                      
                      if len(dir_bef_path) == 0:
                          print("Asking again")
                      else:
                      
                          if dir_bef_path is None:
                              again = True 
                          else:
                              again = False
                              pathRoiStart += dir_bef_path
                              break 
                
                  again = True 
                  
                  while again == True:
                  
                      windowx = sg.Window('Choose path for images after', [[sg.Text('Folder name')], [sg.Input(), sg.FolderBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
                      (keyword, dict_dir) = windowx                
                  
                      dir_after_path = dict_dir['Browse'] 
                      
                      if len(dir_bef_path) == 0:
                          print("Asking again")
                      else:
                      
                          if dir_after_path is None:
                              again = True
                          else:
                              again = False
                              pathRoiEnd += dir_after_path
                              break 
                
                  print("Dir after path: " + dir_after_path)
                  
                  
                  
                  [decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile] = configs_list
                  
                  infi =[decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, pathRoiStart, pathRoiEnd, first_clustering_storing_output, pathPythonFile]
                  
                  infin = [] 
                  
                  for indc, indf in enumerate(infi):
                      indfn = ""
             
                      for ifs in indf:
                          if (ifs is not " ") and (ifs is not "\t") and (ifs is not "\n") and (ifs is not "\r"):
                              indfn += ifs
                              
                      indfn = indfn[1:]
                      
                      if indc == 0 or indc == 2:
                              indfn = int(indfn)
                              
                      infin.append(indfn)
                      
                      print(str(indc) + " - " + str(indfn))
                             
                  
                  infi = infin  
             
                             
              layout = [[sg.ProgressBar(1000, orientation='h', size = (20,20), key = 'progressbar')]]
        
              window = sg.Window('Processing ...', layout)
            
              while True:
                 
                  event, values = window.read(timeout=1000)
                 
                  if event == sg.WIN_CLOSED: 
                      break
                 
                  if True:
                      data_from_tests = []
                      
                      x = 0
             
          ##            infi = (decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile)
                      if '/' in first_clustering_storing_output:
                        x = 1
                      elif "\\" in first_clustering_storing_output:
                        x = 2
                      
                      if os.path.exists(first_clustering_storing_output):
                        if first_clustering_storing_output[-1] == '/' or first_clustering_storing_output[-1] == '//' or first_clustering_storing_output[-1] == "\\":
                            
                            first_clustering_storing_output = first_clustering_storing_output[:-1]
                            first_clustering_storing_output += '_' + "0"
                            
                            ind_newFolder = 1
                            
                            if x == 1:
                                while os.path.exists(first_clustering_storing_output + '/'):
                                   
                                    first_clustering_storing_output[-1] = str(ind_newFolder)
                                    ind_newFolder += 1
                                
                            elif x == 2:
                                while os.path.exists(first_clustering_storing_output + "\\"):
                                    first_clustering_storing_output[-1] = str(ind_newFolder)
                                    ind_newFolder += 1  

                                infi[-2] = first_clustering_storing_output
                       
     ##                 clusteringInfData, executionTime, totCountImages = videoAnalysis(curTest, numberTests, infi, data_from_tests)               
                      
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
                          
                      
                      
                      
                      for i in range(0,3):                      
                          indms = [] 
                          for indm, m in enumerate(mp4_initDir):
                              if m == ':' and indm != 1:
                                  indms.append(indm)
                                  
                          for indm in indms:
                              mp4_initDir = mp4_initDir[:indm] + mp4_initDir[(indm+1):]
                          
                      if not os.path.exists(mp4_initDir):
                          os.mkdir(mp4_initDir) 
                      
                      seqnew = "" 
                      
                      for seqn in str(sequence_name):
                          if seqn != " " and seqn != "\r" and seqn != "\n" and seqn != "\t" and seqn != ":":
                              seqnew += seqn
                              
                      sequence_name_str = seqnew
                      sequence_name = int(sequence_name_str) 
                      
                      path_out = mp4_initDir + "test_" + "00" + str(sequence_name) + '.mp4'
                      path_out_mp4 = pre_dirInit + "test_" + "00" + str(sequence_name) + ".mp4"    ## .mts     
                  
                      print("MP4 file: " + path_out_mp4) 
                      
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
                              print(path_out)
                              src_dir = loadMP4_file(path_in, path_out_mp4) 
                              print("path_out_mp4: \t" + path_out_mp4)
                              print("path_out: \t" + path_out)
                              time.sleep(100)  
                              if not os.path.exists(path_out):
                                  os.rename(path_out_mp4, path_out) 
                                  while(os.path.exists(path_out) == False): 
                                      time.sleep(5)
                                  print("MP4 file loaded")          
                          
                          time.sleep(100)    
                          
                          src_dir = path_out  
                          
                      output_video_filenames, frame_rate, n_images, buffer_imgs, n_parts = sepVideosFromWholeOne(True, infi[5] + infi[7] + "\\" + "test_000.mp4")    ## (False, None)
                      
                      if output_video_filenames is None:
                          
                     #     print(output_video_filenames, frame_rate, n_images, buffer_imgs)
                          
                          # import sys 
                          # sys.exit()
                          
                          if n_parts == 2:
                              
                              featuresData, metrics_names = videoAnalysis_single_justFeatures(infi, newPath, n_images)
                           
                          # data_from_tests = videoAnalysisJustFeatures(curTest, numberTests, infi, data_from_tests) 
                 
                          # featuresData = videoAnalysisJustFeatures(curTest, numberTests, infi, data_from_tests)
                              gui_show_results(featuresData, metrics_names)
                          
                          break  
                      else:
                          print("Going to execute a software for only a video, but analysing each core moment")
                          ok = False
                         ## (curTest, numberTests, tupleForProcessing, data_from_tests, direcPythonFile, filename_output_video):
                           
                          curTest = 0
                          limToCompare = 50
                          numberTests = len(output_video_filenames)  
                          
                          numberTests -= 1
                          
                          while True:       
                                          
                            
                                         if os.path.isfile('temp_numberTests.txt'):
                                             numberTests = read_number_tests_from_file()
                              
                                         print("CurTest: " + str(curTest))
                                         print("Total tests: " + str(numberTests))
                                         
                                         
                             ##             if ind_inf != len(infx) - 1: 
                                         if curTest < numberTests-1:
                                              
                                             print("Inside")  
                                               
                                             
                                             if '/' in first_clustering_storing_output:
                                                 x = 1
                                             elif "\\" in first_clustering_storing_output:
                                                 x = 2
                                               
                                             if os.path.exists(first_clustering_storing_output):
                                                 if first_clustering_storing_output[-1] == '/' or first_clustering_storing_output[-1] == '//' or first_clustering_storing_output[-1] == "\\":
                                                     
                                                     first_clustering_storing_output = first_clustering_storing_output[:-1]
                                                     first_clustering_storing_output += '_' + "0"
                                                     
                                                     ind_newFolder = 1  
                                                     
                                                     if x == 1:
                                                         if not os.path.exists(first_clustering_storing_output + '/'):
                                                             os.mkdir(first_clustering_storing_output + '/')
                                                         while os.path.exists(first_clustering_storing_output + '/'):                                               
                                                             first_clustering_storing_output[-1] = str(ind_newFolder)
                                                             ind_newFolder += 1
                                                         else:
                                                             os.mkdir(first_clustering_storing_output + '/')
                                                         
                                                     elif x == 2:
                                                         if not os.path.exists(first_clustering_storing_output + "\\"):
                                                             os.mkdir(first_clustering_storing_output + "\\")
                                                         while os.path.exists(first_clustering_storing_output + "\\"):
                                                             first_clustering_storing_output[-1] = str(ind_newFolder)
                                                             ind_newFolder += 1 
                                                         else:
                                                             os.mkdir(first_clustering_storing_output + "\\")

                                                 infi[-2] = first_clustering_storing_output
                                            
                                             print("Length for infi before: " + str(len(infi)))
                                             print(infi)
                                             print("------------------------")
                                             
                                             if len(infi) == 14:
                                             
                                                 infis = infi[:-4]
                                                 infis.append(infi[-2])
                                                 infis.append(infi[-1])
                                                 infi = infis
                                             
                                                 print("Infi: ")
                                                 print(infi)
                                                 print("Length for infi: " + str(len(infi)))                                          
                                           
                                             limToCompare = np.min(np.array([n_images]))
                                             print("Limite to compare: " + str(limToCompare))
                                             
                                             print(output_video_filenames[curTest])
                                             
                                             indSepx=[]
                                             for indx, x in enumerate(dir_after_path):
                                                    if x == '/':
                                                        indSepx.append(indx)
                                                
                                             indSep = indSepx[-2]                                        
                                           
                                             dirVideo = dir_after_path[:indSep] + '/' + output_video_filenames[curTest]
                                             
                                             data_from_tests = videoAnalysisMultipleVideosWithinOneJustFeatures(curTest, numberTests, infi, data_from_tests, this_dir, dirVideo, frame_rate, limToCompare, buffer_imgs)
                                               
                                             print("Length for data_from_tests: " + str(len(data_from_tests)))
                                             
                                             if len(data_from_tests) > 0: 
                                                 for d in data_from_tests:
                                                     print("Data: ")
                                                     for x in d:  
                                                         print(str(x) + " \t")   
                                             
                                             curTest += 1  
                ##                             infx += 1 
                                             
                                             if curTest == numberTests-1:
                                                print("Turn")
                                         #       break
                                              
                                         else:
                                             
                                             print("Passed") 
                                             
                                             print("----------------------------")
                                             print("----------------------------")
                                             print("----------------------------")
                                             print("----------------------------")
                                             print("----------------------------")
                                          
                                             if '/' in first_clustering_storing_output:
                                                 x = 1
                                             elif "\\" in first_clustering_storing_output:
                                                 x = 2
                                                  
                                             if os.path.exists(first_clustering_storing_output):
                                                 if first_clustering_storing_output[-1] == '/' or first_clustering_storing_output[-1] == '//' or first_clustering_storing_output[-1] == "\\":
                                                     
                                                     first_clustering_storing_output = first_clustering_storing_output[:-1]
                                                     first_clustering_storing_output += '_' + "0"
                                                     
                                                     ind_newFolder = 1 
                                                       
                                                     if x == 1:
                                                         if not os.path.exists(first_clustering_storing_output + '/'):
                                                             os.mkdir(first_clustering_storing_output + '/')
                                                         while os.path.exists(first_clustering_storing_output + '/'):                                               
                                                             first_clustering_storing_output[-1] = str(ind_newFolder)
                                                             ind_newFolder += 1
                                                         else:
                                                             os.mkdir(first_clustering_storing_output + '/')
                                                          
                                                         
                                                     elif x == 2:
                                                         if not os.path.exists(first_clustering_storing_output + "\\"):
                                                             os.mkdir(first_clustering_storing_output + "\\")
                                                         while os.path.exists(first_clustering_storing_output + "\\"):
                                                             first_clustering_storing_output[-1] = str(ind_newFolder)
                                                             ind_newFolder += 1  
                                                         else:
                                                             os.mkdir(first_clustering_storing_output + "\\")

                                                 infi[-2] = first_clustering_storing_output
                                                 
                                           ##      print("Current test here: " + str(curTest))
                                           ##      print("Number of tests: " + str(numberTests))
                                                 print("Length here: " + str(len(data_from_tests)))                                            
                                                 
                                                 if len(infi) == 14:
                                                      
                                                     infis = infi[:-4]
                                                     infis.append(infi[-2])
                                                     infis.append(infi[-1])
                                                     infi = infis
                                                     
                                                     for d in data_from_tests:
                                                         print(d)
                                                
                                                 ok = True
                                                 
                                                 print("\n\n CurTest for last one: " + str(curTest))
                                                 print("\n\n numberTests: " + str(numberTests) + "\n\n")
                                                 
                                                 ################################
                                                 
                                                 print(output_video_filenames[curTest])
                                                 
                                                 indSepx=[]
                                                 for indx, x in enumerate(dir_after_path):
                                                     if x == '/':
                                                         indSepx.append(indx)
                                                 
                                                 indSep = indSepx[-2]
                                               
                                                 dirVideo = dir_after_path[:indSep] + '/' + output_video_filenames[curTest]
                                                 
                                                  
                                                 featuresData, metrics_names = videoAnalysisMultipleVideosWithinOneJustFeatures(curTest, numberTests, infi, data_from_tests, this_dir, dirVideo, frame_rate, limToCompare, buffer_imgs)
                                                 
                                                 if featuresData is None:
                                                     print("Check features info")
                                                     raise Exception
                                                 else:
                                                     print("Features ok")
                                            
                                             gui_show_results(featuresData, metrics_names) 
                                                     
                                                  
                          break 
                
                
              window.close()
              
              ################################
             
              # break
              
        else:
            
                ind_inf = 0
        
        
        
      #       if numberTests == 1:
                  
      #           if curTest == 0:
      #               infx = []
      #               data_to_save = []
                     
      #    ##       decisorLevel = 0
                
      #           [decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile] = infi
                
                
      #           infx.append((decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile))
                
      # ##          nextTest = countdown_timer_display(time_bet_tests)    ## in minutes         
      #           nextTest = True
      #           print("Ready for next test: " + str(nextTest))
                
 ##           else:  
    
                startTime = time.time()             
                
               
                data_from_tests = []
                data_to_save = []
               
          #      for ind_inf, infi in enumerate(infx):
              
                if True:
                    
                    layout = [[sg.ProgressBar(1000, orientation='h', size = (20,20), key = 'progressbar')]]
        
                    window = sg.Window('Processing ...', layout)
                   
                    while True: 
                        
                        event, values = window.read(timeout=1000)
                        
                        if event == sg.WIN_CLOSED:
                            break
                         
                        while True:
                            
                            if True:      
                                
                                again = True
                                
                                dir_configs_path = config_dirs[ind_inf]
                                
                                # while again == True:
                                 
                                #     windowx = sg.Window('Choose path to configs file', [[sg.Text('File name')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
                                #     (keyword, dict_dir) = windowx                
                                
                                #     dir_configs_path = dict_dir['Browse'] 
                                    
                                #     if dir_configs_path is None:
                                #         again = True
                                #     else:
                                #         if not ".txt" in dir_configs_path:
                                #             again = True
                                #         else:
                                #             again = False
                                #             break
                                
                                print(dir_configs_path)
                                 
                                configs_list = read_configs_file(dir_configs_path)    
                                
                                [decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile] = configs_list
                                infi =[decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile]
                                
                                infin = [] 
                                
                                for indc, indf in enumerate(infi):
                                    indfn = ""
                           
                                    for ifs in indf:
                                        if (ifs is not " ") and (ifs is not "\t") and (ifs is not "\n") and (ifs is not "\r"):
                                            indfn += ifs
                                            
                                    indfn = indfn[1:]
                                    
                                    if indc == 0 or indc == 2:
                                            indfn = int(indfn)
                                            
                                    infin.append(indfn)
                                    
                                    print(str(indc) + " - " + str(indfn))
                                           
                                
                                infi = infin  
                            
                            infx.append((decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile))
                            infi = (decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile)
                            
                            print("Looking for info " + str(ind_inf) + " ...")
                            print("Length infx: " + str(len(infx)))
                            print("Current test: " + str(curTest))
                            print("Number tests: " + str(numberTests))
                            
                            if curTest < numberTests-1:
                                time.sleep(10)
                              
                ##             if ind_inf != len(infx) - 1: 
                            
                                 
                                print("Inside")  
                                
                                 
                                if '/' in first_clustering_storing_output:
                                    x = 1
                                elif "\\" in first_clustering_storing_output:
                                    x = 2
                                  
                                if os.path.exists(first_clustering_storing_output):
                                    if first_clustering_storing_output[-1] == '/' or first_clustering_storing_output[-1] == '//' or first_clustering_storing_output[-1] == "\\":
                                        
                                        first_clustering_storing_output = first_clustering_storing_output[:-1]
                                        first_clustering_storing_output += '_' + "0"
                                        
                                        ind_newFolder = 1 
                                        
                                        if x == 1:
                                            if not os.path.exists(first_clustering_storing_output + '/'):
                                                os.mkdir(first_clustering_storing_output + '/')
                                            while os.path.exists(first_clustering_storing_output + '/'):                                               
                                                first_clustering_storing_output[-1] = str(ind_newFolder)
                                                ind_newFolder += 1
                                            else:
                                                os.mkdir(first_clustering_storing_output + '/')
                                            
                                        elif x == 2:
                                            if not os.path.exists(first_clustering_storing_output + "\\"):
                                                os.mkdir(first_clustering_storing_output + "\\")
                                            while os.path.exists(first_clustering_storing_output + "\\"):
                                                first_clustering_storing_output[-1] = str(ind_newFolder)
                                                ind_newFolder += 1 
                                            else:
                                                os.mkdir(first_clustering_storing_output + "\\")

                                    infi[-2] = first_clustering_storing_output
                    
                                data_from_tests = videoAnalysisJustFeatures(curTest, numberTests, infi, data_from_tests) 
                                  
                                print("Length for data_from_tests: " + str(len(data_from_tests)))
                                
                                if len(data_from_tests) > 0: 
                                    for d in data_from_tests:
                                        print("Data: ")
                                        for x in d:  
                                            print(str(x) + " \t")  
                                
                                curTest += 1  
   ##                             infx += 1 
                                
                                if curTest == numberTests-1:
                                   print("Turn")
                            #       break
                                
                            else:
                                
                                print("Passed")
                                
                                print("----------------------------")
                                print("----------------------------")
                                print("----------------------------")
                                print("----------------------------")
                                print("----------------------------")
                             
                                if '/' in first_clustering_storing_output:
                                    x = 1
                                elif "\\" in first_clustering_storing_output:
                                    x = 2
                                     
                                print("--A")
                                
                                
                                     
                                if os.path.exists(first_clustering_storing_output):
                                    print("--B")
                                    if first_clustering_storing_output[-1] == '/' or first_clustering_storing_output[-1] == '//' or first_clustering_storing_output[-1] == "\\":
                                        print("--C")
                                        
                                        first_clustering_storing_output = first_clustering_storing_output[:-1]
                                        first_clustering_storing_output += '_' + "0"
                                        
                                        ind_newFolder = 1  
                                         
                                        print("--D")
                                           
                                        if x == 1:
                                            if not os.path.exists(first_clustering_storing_output + '/'):
                                                os.mkdir(first_clustering_storing_output + '/')
                                            while os.path.exists(first_clustering_storing_output + '/'):                                               
                                                first_clustering_storing_output[-1] = str(ind_newFolder)
                                                ind_newFolder += 1
                                            else:
                                                os.mkdir(first_clustering_storing_output + '/')
                                             
                                            
                                        elif x == 2:
                                            if not os.path.exists(first_clustering_storing_output + "\\"):
                                                os.mkdir(first_clustering_storing_output + "\\")
                                            while os.path.exists(first_clustering_storing_output + "\\"):
                                                first_clustering_storing_output[-1] = str(ind_newFolder)
                                                ind_newFolder += 1  
                                            else:
                                                os.mkdir(first_clustering_storing_output + "\\")
                                        print("--E")
                                        
                                    infi[-2] = first_clustering_storing_output
                                
                                if True:
                          ##          break 
                                    
                          ##      print("Current test here: " + str(curTest))
                          ##      print("Number of tests: " + str(numberTests))
                                    print("Length here: " + str(len(data_from_tests)))
                                    print("Info going in: ")
                                    print(curTest)
                                    print(numberTests)
                                    print(infi)
                                    print(data_from_tests)
                                    featuresData, metrics_names = videoAnalysisJustFeatures(curTest, numberTests, infi, data_from_tests) 
                                    
                            #        print("Features Data: ")
                           #         print(featuresData)
                           
                                    data_to_save.append(featuresData)
                                    
                                    print("Metrics names: ")
                                    print(metrics_names)
                                    
                                    if featuresData is None:
                                        print("Check features info")
                                        raise Exception
                                    else:
                                        print("Features ok")
                                   
                                    break 
                                
                                break
                                        
                                    
                                    # print("C")     
                                            
                            if completed:
                                break
                                    
                                     
                                    
                                    # if completed:                            
                                    #     break
                                    # else:
                                    #     print("NO DATA AVAILABLE")
                                    #     break
                    
                            ind_inf += 1
                        break
                    
                    print("D")
                    window.close() 
                 
      #          counterTest += 1 
  ##              print("Waiting to see what happens ...")
##                cv2.waitKey(0)

                print("Data:")
                # for d in data_to_save:
                #     for x in d:                        
                #         print(x)   
                print(data_to_save)
                
                print("Length for data to save: " + str(len(data_to_save)))  
                
           #     if not isinstance(data_to_save[0], int) and not isinstance(data_to_save[0], float):
                   
           #         print("Length for data to save: " + str(len(data_to_save[0])))  
                
                
         #           if len(data_to_save[0]) > 0: 
                         
                         # if len(data_to_save) != 3: 
                         #     print("Check length data ...")
                         # else:                                       
               
                         # for ind_data, data in enumerate(data_to_save):
                             
                         #         print(data)
                         #         print("Length of Data after: " + str(len(data))) 
                                
                                
                         #         if len(data) == 3:
                
                for featuresData in data_to_save:
                                         
                    gui_show_results(featuresData, metrics_names) 
                                     
                                     # if ind_data != len(data_to_save)-1:
                                     #     print(" -- Showing GUI with results for test number " + str(ind_data+1))
                                     #     time.sleep(5)
                                     # else:
                                     #     print("Terminating ...")
                               
                                    
                    # else:
                    #      print("Output results not available !!!")                  
           
        
                        ###################
        
                executionTime = (time.time() - startTime)
                print('Whole execution time in seconds: ' + str(executionTime))
                      
                time.sleep(5)    
                print("-- Done")
                
                sys.exit()
         
     #           break 
           
        
            
## post_proc_pfs_only()